#!/usr/bin/env python3
"""
Ray-based Distributed Processing Manager for Multi-GPU S2V System
Handles distributed workload management, GPU allocation, and result aggregation
"""

import ray
import time
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import psutil
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """Data class for processing tasks"""
    task_id: str
    chunk_data: Dict
    priority: int = 0
    estimated_processing_time: float = 0.0
    gpu_memory_required: float = 0.0


@dataclass
class WorkerStats:
    """Worker performance statistics"""
    worker_id: int
    gpu_id: int
    total_tasks_processed: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    current_memory_usage: float = 0.0
    peak_memory_usage: float = 0.0
    error_count: int = 0
    last_task_time: Optional[float] = None


class GPUResourceManager:
    """Manages GPU resources and allocation"""
    
    def __init__(self):
        self.gpu_info = self._get_gpu_info()
        self.allocation_lock = threading.Lock()
        self.allocated_memory = {i: 0.0 for i in range(len(self.gpu_info))}
    
    def _get_gpu_info(self) -> List[Dict]:
        """Get detailed GPU information"""
        if not torch.cuda.is_available():
            return []
        
        gpu_list = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_list.append({
                'id': i,
                'name': props.name,
                'total_memory': props.total_memory / 1024**3,  # GB
                'compute_capability': f"{props.major}.{props.minor}",
                'multiprocessor_count': props.multiprocessor_count
            })
        
        return gpu_list
    
    def get_available_gpus(self, memory_required: float = 0.0) -> List[int]:
        """Get list of available GPUs with sufficient memory"""
        available = []
        
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                available_memory = gpu.memoryTotal - gpu.memoryUsed
                if available_memory >= memory_required * 1024:  # Convert GB to MB
                    available.append(gpu.id)
        except Exception as e:
            logger.warning(f"Could not get GPU utilization: {e}")
            # Fallback to CUDA device count
            available = list(range(torch.cuda.device_count()))
        
        return available
    
    def allocate_gpu(self, memory_required: float) -> Optional[int]:
        """Allocate a GPU for processing"""
        with self.allocation_lock:
            available_gpus = self.get_available_gpus(memory_required)
            
            if not available_gpus:
                return None
            
            # Select GPU with least allocated memory
            best_gpu = min(available_gpus, 
                         key=lambda gpu_id: self.allocated_memory.get(gpu_id, 0))
            
            self.allocated_memory[best_gpu] += memory_required
            return best_gpu
    
    def release_gpu(self, gpu_id: int, memory_released: float):
        """Release GPU memory allocation"""
        with self.allocation_lock:
            if gpu_id in self.allocated_memory:
                self.allocated_memory[gpu_id] = max(0, 
                    self.allocated_memory[gpu_id] - memory_released)


@ray.remote
class WorkerMonitor:
    """Ray actor to monitor worker performance"""
    
    def __init__(self):
        self.worker_stats = {}
        self.system_stats = []
        self.start_time = time.time()
    
    def update_worker_stats(self, worker_id: int, stats_update: Dict):
        """Update worker statistics"""
        if worker_id not in self.worker_stats:
            self.worker_stats[worker_id] = WorkerStats(
                worker_id=worker_id,
                gpu_id=stats_update.get('gpu_id', -1)
            )
        
        worker_stat = self.worker_stats[worker_id]
        
        # Update counters
        if 'processing_time' in stats_update:
            worker_stat.total_tasks_processed += 1
            worker_stat.total_processing_time += stats_update['processing_time']
            worker_stat.average_processing_time = (
                worker_stat.total_processing_time / worker_stat.total_tasks_processed
            )
            worker_stat.last_task_time = time.time()
        
        # Update memory usage
        if 'memory_usage' in stats_update:
            worker_stat.current_memory_usage = stats_update['memory_usage']
            worker_stat.peak_memory_usage = max(
                worker_stat.peak_memory_usage,
                stats_update['memory_usage']
            )
        
        # Update error count
        if 'error' in stats_update:
            worker_stat.error_count += 1
    
    def get_all_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # System resources
        system_info = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'uptime_seconds': uptime
        }
        
        # GPU information
        gpu_info = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': gpu.memoryUtil * 100,
                    'gpu_util': gpu.load * 100,
                    'temperature': gpu.temperature
                })
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")
        
        return {
            'worker_stats': {k: v.__dict__ for k, v in self.worker_stats.items()},
            'system_info': system_info,
            'gpu_info': gpu_info,
            'total_workers': len(self.worker_stats),
            'uptime_seconds': uptime
        }
    
    def log_system_stats(self):
        """Log current system statistics"""
        stats = self.get_all_stats()
        logger.info(f"System Stats - CPU: {stats['system_info']['cpu_percent']:.1f}%, "
                   f"Memory: {stats['system_info']['memory_percent']:.1f}%, "
                   f"Workers: {stats['total_workers']}")


class DistributedTaskScheduler:
    """Intelligent task scheduler for distributed processing"""
    
    def __init__(self, workers: List[ray.ObjectRef], monitor: ray.ObjectRef):
        self.workers = workers
        self.monitor = monitor
        self.task_queue = queue.PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.resource_manager = GPUResourceManager()
        
    def submit_task(self, task: ProcessingTask) -> str:
        """Submit a task for processing"""
        # Estimate GPU memory requirement if not provided
        if task.gpu_memory_required == 0.0:
            task.gpu_memory_required = self._estimate_memory_requirement(task.chunk_data)
        
        # Add to priority queue (negative priority for max-heap behavior)
        self.task_queue.put((-task.priority, task.task_id, task))
        
        logger.debug(f"Task {task.task_id} submitted with priority {task.priority}")
        return task.task_id
    
    def _estimate_memory_requirement(self, chunk_data: Dict) -> float:
        """Estimate GPU memory requirement for a task"""
        # Simple estimation based on chunk parameters
        num_frames = chunk_data.get('num_frames', 25)
        height = chunk_data.get('height', 480)
        width = chunk_data.get('width', 640)
        
        # Rough estimate: base model memory + frame data
        base_memory = 2.0  # GB for model
        frame_memory = (num_frames * height * width * 3 * 4) / (1024**3)  # RGB float32
        
        return base_memory + frame_memory * 2  # 2x for processing overhead
    
    def process_tasks_parallel(self, max_concurrent: int = None) -> Dict:
        """Process all tasks in parallel"""
        if max_concurrent is None:
            max_concurrent = len(self.workers)
        
        results = {}
        active_futures = {}
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            while not self.task_queue.empty() or active_futures:
                # Submit new tasks if we have capacity
                while (len(active_futures) < max_concurrent and 
                       not self.task_queue.empty()):
                    
                    try:
                        _, task_id, task = self.task_queue.get_nowait()
                        
                        # Find available worker
                        worker_id = self._find_available_worker(task.gpu_memory_required)
                        if worker_id is not None:
                            future = executor.submit(self._process_single_task, 
                                                   worker_id, task)
                            active_futures[future] = (task_id, task, worker_id)
                            self.active_tasks[task_id] = task
                        else:
                            # No available worker, put task back
                            self.task_queue.put((-task.priority, task_id, task))
                            time.sleep(0.1)  # Brief pause before retrying
                            break
                    
                    except queue.Empty:
                        break
                
                # Check for completed tasks
                completed = []
                for future in list(active_futures.keys()):
                    if future.done():
                        completed.append(future)
                
                # Process completed tasks
                for future in completed:
                    task_id, task, worker_id = active_futures.pop(future)
                    
                    try:
                        result = future.result()
                        results[task_id] = result
                        self.completed_tasks[task_id] = task
                        
                        # Update monitor
                        ray.get(self.monitor.update_worker_stats.remote(
                            worker_id, {
                                'processing_time': result.get('processing_time', 0),
                                'memory_usage': result.get('memory_usage', 0)
                            }
                        ))
                        
                    except Exception as e:
                        logger.error(f"Task {task_id} failed: {e}")
                        self.failed_tasks[task_id] = {'task': task, 'error': str(e)}
                        
                        # Update error count
                        ray.get(self.monitor.update_worker_stats.remote(
                            worker_id, {'error': True}
                        ))
                    
                    finally:
                        # Release GPU resources
                        self.resource_manager.release_gpu(
                            worker_id, task.gpu_memory_required
                        )
                        
                        if task_id in self.active_tasks:
                            del self.active_tasks[task_id]
                
                # Brief pause to prevent busy waiting
                if not active_futures and not self.task_queue.empty():
                    time.sleep(0.1)
        
        return {
            'results': results,
            'completed_count': len(self.completed_tasks),
            'failed_count': len(self.failed_tasks),
            'failed_tasks': self.failed_tasks
        }
    
    def _find_available_worker(self, memory_required: float) -> Optional[int]:
        """Find an available worker with sufficient resources"""
        available_gpus = self.resource_manager.get_available_gpus(memory_required)
        
        if not available_gpus:
            return None
        
        # Simple round-robin selection among available workers
        for i, worker in enumerate(self.workers):
            if i in available_gpus:
                gpu_id = self.resource_manager.allocate_gpu(memory_required)
                if gpu_id is not None:
                    return i
        
        return None
    
    def _process_single_task(self, worker_id: int, task: ProcessingTask) -> Dict:
        """Process a single task using specified worker"""
        worker = self.workers[worker_id]
        
        # Submit task to worker
        future = worker.process_chunk.remote(
            task.chunk_data['image_data'],
            task.chunk_data['audio_features'],
            task.chunk_data['prompt'],
            {k: v for k, v in task.chunk_data.items() 
             if k not in ['image_data', 'audio_features', 'prompt']}
        )
        
        # Wait for result
        result = ray.get(future)
        result['task_id'] = task.task_id
        result['worker_id'] = worker_id
        
        return result


class LoadBalancer:
    """Dynamic load balancer for optimal resource utilization"""
    
    def __init__(self, monitor: ray.ObjectRef):
        self.monitor = monitor
        self.load_history = []
        self.rebalance_threshold = 0.2  # 20% load imbalance triggers rebalancing
    
    def analyze_load_distribution(self) -> Dict:
        """Analyze current load distribution across workers"""
        stats = ray.get(self.monitor.get_all_stats.remote())
        worker_stats = stats['worker_stats']
        
        if not worker_stats:
            return {'balanced': True, 'recommendations': []}
        
        # Calculate load metrics
        processing_times = []
        memory_usage = []
        
        for worker_stat in worker_stats.values():
            processing_times.append(worker_stat['average_processing_time'])
            memory_usage.append(worker_stat['current_memory_usage'])
        
        # Analyze balance
        if processing_times:
            time_std = np.std(processing_times)
            time_mean = np.mean(processing_times)
            time_imbalance = time_std / max(time_mean, 0.001)
        else:
            time_imbalance = 0
        
        if memory_usage:
            memory_std = np.std(memory_usage)
            memory_mean = np.mean(memory_usage)
            memory_imbalance = memory_std / max(memory_mean, 0.001)
        else:
            memory_imbalance = 0
        
        # Generate recommendations
        recommendations = []
        if time_imbalance > self.rebalance_threshold:
            recommendations.append(
                f"Processing time imbalance detected ({time_imbalance:.2f}). "
                "Consider redistributing tasks."
            )
        
        if memory_imbalance > self.rebalance_threshold:
            recommendations.append(
                f"Memory usage imbalance detected ({memory_imbalance:.2f}). "
                "Consider memory optimization."
            )
        
        return {
            'balanced': time_imbalance <= self.rebalance_threshold and 
                       memory_imbalance <= self.rebalance_threshold,
            'time_imbalance': time_imbalance,
            'memory_imbalance': memory_imbalance,
            'recommendations': recommendations,
            'worker_count': len(worker_stats)
        }
    
    def suggest_optimal_chunk_size(self, total_frames: int, num_workers: int) -> int:
        """Suggest optimal chunk size based on current system performance"""
        # Get recent performance data
        stats = ray.get(self.monitor.get_all_stats.remote())
        worker_stats = stats['worker_stats']
        
        if not worker_stats:
            # Default chunk size
            return max(10, total_frames // num_workers)
        
        # Calculate average processing time per frame
        total_processing_time = sum(w['total_processing_time'] for w in worker_stats.values())
        total_tasks = sum(w['total_tasks_processed'] for w in worker_stats.values())
        
        if total_tasks == 0:
            return max(10, total_frames // num_workers)
        
        avg_time_per_task = total_processing_time / total_tasks
        
        # Target processing time per chunk (aim for ~30 seconds per chunk)
        target_chunk_time = 30.0
        suggested_chunk_size = max(10, int(target_chunk_time / avg_time_per_task))
        
        # Ensure reasonable bounds
        min_chunk_size = max(5, total_frames // (num_workers * 4))  # At most 4x workers chunks
        max_chunk_size = total_frames // num_workers if num_workers > 0 else total_frames
        
        suggested_chunk_size = max(min_chunk_size, 
                                  min(suggested_chunk_size, max_chunk_size))
        
        logger.info(f"Suggested chunk size: {suggested_chunk_size} frames "
                   f"(based on {avg_time_per_task:.2f}s avg per task)")
        
        return suggested_chunk_size