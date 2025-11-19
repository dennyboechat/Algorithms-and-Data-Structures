"""
Performance Monitoring and Profiling System

This module provides comprehensive performance monitoring, profiling, and 
benchmarking capabilities for the optimized recommendation system, enabling
detailed analysis of computational efficiency, memory usage, and scalability.

Features:
- Real-time performance monitoring
- Memory profiling and leak detection
- CPU usage analysis
- Algorithm-specific profiling
- Comparative performance analysis
- Automatic performance regression detection
- Detailed reporting and visualization

References:
- Goldberg, D., et al. (1992). Using collaborative filtering to weave an 
  information tapestry. Communications of the ACM, 35(12), 61-70.
- Miller, B. P., et al. (1995). The Paradyn parallel performance measurement tool. 
  Computer, 28(11), 37-46.
"""

import time
import threading
import gc
import os
import sys
import traceback
import functools
import cProfile
import pstats
import io
from typing import Dict, List, Any, Optional, Callable, Union
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import psutil
    import resource
    SYSTEM_MONITORING_AVAILABLE = True
except ImportError:
    SYSTEM_MONITORING_AVAILABLE = False
    print("Warning: psutil not available. System monitoring features disabled.")

try:
    import numpy as np
    import pandas as pd
    NUMPY_PANDAS_AVAILABLE = True
except ImportError:
    NUMPY_PANDAS_AVAILABLE = False
    print("Warning: numpy/pandas not available. Some analysis features disabled.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Visualization disabled.")


class PerformanceMonitor:
    """
    Real-time performance monitoring system.
    
    Tracks various performance metrics in real-time and provides
    alerts when thresholds are exceeded.
    """
    
    def __init__(self, monitoring_interval: float = 1.0, 
                 alert_thresholds: Dict[str, float] = None):
        """
        Initialize performance monitor.
        
        Args:
            monitoring_interval: Interval between measurements (seconds)
            alert_thresholds: Thresholds for performance alerts
        """
        self.monitoring_interval = monitoring_interval
        self.alert_thresholds = alert_thresholds or {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'response_time': 5.0,
            'error_rate': 0.05
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Performance data storage
        self.performance_data = defaultdict(deque)
        self.alerts = []
        self.max_data_points = 1000
        
        # Metrics tracking
        self.operation_counts = defaultdict(int)
        self.operation_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # System information
        if SYSTEM_MONITORING_AVAILABLE:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        if self.is_monitoring:
            return
        
        print("Starting performance monitoring...")
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        if not self.is_monitoring:
            return
        
        print("Stopping performance monitoring...")
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                timestamp = time.time()
                metrics = self._collect_system_metrics()
                
                # Store metrics
                with self.lock:
                    for metric_name, value in metrics.items():
                        self.performance_data[metric_name].append((timestamp, value))
                        
                        # Keep only recent data points
                        while len(self.performance_data[metric_name]) > self.max_data_points:
                            self.performance_data[metric_name].popleft()
                
                # Check for alerts
                self._check_alerts(metrics, timestamp)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system performance metrics."""
        metrics = {}
        
        if SYSTEM_MONITORING_AVAILABLE and self.process:
            try:
                # CPU usage
                metrics['cpu_percent'] = self.process.cpu_percent()
                
                # Memory usage
                memory_info = self.process.memory_info()
                metrics['memory_rss_mb'] = memory_info.rss / 1024 / 1024
                metrics['memory_vms_mb'] = memory_info.vms / 1024 / 1024
                
                # Memory percentage
                system_memory = psutil.virtual_memory()
                metrics['memory_percent'] = (memory_info.rss / system_memory.total) * 100
                
                # Thread count
                metrics['thread_count'] = self.process.num_threads()
                
                # File descriptors (on Unix systems)
                try:
                    metrics['open_files'] = self.process.num_fds()
                except (AttributeError, psutil.AccessDenied):
                    pass
                
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"Error collecting system metrics: {e}")
        
        # Python-specific metrics
        metrics['active_threads'] = threading.active_count()
        
        # Garbage collection stats
        gc_stats = gc.get_stats()
        if gc_stats:
            for i, stat in enumerate(gc_stats):
                metrics[f'gc_gen{i}_collections'] = stat['collections']
                metrics[f'gc_gen{i}_collected'] = stat['collected']
        
        return metrics
    
    def _check_alerts(self, metrics: Dict[str, float], timestamp: float) -> None:
        """Check if any metrics exceed alert thresholds."""
        for metric_name, value in metrics.items():
            if metric_name in self.alert_thresholds:
                threshold = self.alert_thresholds[metric_name]
                if value > threshold:
                    alert = {
                        'timestamp': timestamp,
                        'metric': metric_name,
                        'value': value,
                        'threshold': threshold,
                        'severity': 'warning' if value < threshold * 1.2 else 'critical'
                    }
                    self.alerts.append(alert)
                    print(f"ALERT: {metric_name} = {value:.2f} (threshold: {threshold})")
    
    @contextmanager
    def operation_timer(self, operation_name: str):
        """
        Context manager for timing operations.
        
        Usage:
            with monitor.operation_timer('algorithm_training'):
                # code to time
                pass
        """
        start_time = time.perf_counter()
        error_occurred = False
        
        try:
            yield
        except Exception as e:
            error_occurred = True
            with self.lock:
                self.error_counts[operation_name] += 1
            raise
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            with self.lock:
                self.operation_counts[operation_name] += 1
                self.operation_times[operation_name].append(execution_time)
                
                # Store in performance data
                timestamp = time.time()
                self.performance_data[f'{operation_name}_time'].append((timestamp, execution_time))
                
                # Keep operation times list manageable
                if len(self.operation_times[operation_name]) > 1000:
                    self.operation_times[operation_name] = self.operation_times[operation_name][-500:]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self.lock:
            current_metrics = {}
            
            # Get latest values for each metric
            for metric_name, data_points in self.performance_data.items():
                if data_points:
                    current_metrics[metric_name] = data_points[-1][1]
            
            # Operation statistics
            current_metrics['operation_stats'] = {}
            for op_name, count in self.operation_counts.items():
                times = self.operation_times[op_name]
                errors = self.error_counts[op_name]
                
                if times:
                    current_metrics['operation_stats'][op_name] = {
                        'count': count,
                        'avg_time': sum(times) / len(times),
                        'min_time': min(times),
                        'max_time': max(times),
                        'error_count': errors,
                        'error_rate': errors / count if count > 0 else 0.0
                    }
        
        return current_metrics
    
    def get_performance_summary(self, time_window: int = 3600) -> Dict[str, Any]:
        """
        Get performance summary for a time window.
        
        Args:
            time_window: Time window in seconds (default: 1 hour)
            
        Returns:
            Performance summary statistics
        """
        current_time = time.time()
        start_time = current_time - time_window
        
        summary = {
            'time_window_seconds': time_window,
            'metrics_summary': {},
            'operation_summary': {},
            'alerts_summary': {}
        }
        
        with self.lock:
            # Summarize metric data within time window
            for metric_name, data_points in self.performance_data.items():
                window_data = [value for timestamp, value in data_points 
                             if timestamp >= start_time]
                
                if window_data and NUMPY_PANDAS_AVAILABLE:
                    summary['metrics_summary'][metric_name] = {
                        'count': len(window_data),
                        'mean': np.mean(window_data),
                        'median': np.median(window_data),
                        'std': np.std(window_data),
                        'min': np.min(window_data),
                        'max': np.max(window_data),
                        'percentile_95': np.percentile(window_data, 95)
                    }
                elif window_data:
                    # Basic statistics without numpy
                    summary['metrics_summary'][metric_name] = {
                        'count': len(window_data),
                        'mean': sum(window_data) / len(window_data),
                        'min': min(window_data),
                        'max': max(window_data)
                    }
            
            # Operation summary
            for op_name in self.operation_counts:
                times = self.operation_times[op_name]
                window_times = [t for t in times if True]  # Simplified for now
                
                if window_times and NUMPY_PANDAS_AVAILABLE:
                    summary['operation_summary'][op_name] = {
                        'total_operations': self.operation_counts[op_name],
                        'avg_time': np.mean(window_times),
                        'median_time': np.median(window_times),
                        'p95_time': np.percentile(window_times, 95),
                        'error_count': self.error_counts[op_name],
                        'throughput_ops_per_sec': len(window_times) / time_window if time_window > 0 else 0
                    }
            
            # Alert summary
            window_alerts = [alert for alert in self.alerts 
                           if alert['timestamp'] >= start_time]
            
            alert_counts = defaultdict(int)
            for alert in window_alerts:
                alert_counts[alert['severity']] += 1
            
            summary['alerts_summary'] = {
                'total_alerts': len(window_alerts),
                'by_severity': dict(alert_counts),
                'recent_alerts': window_alerts[-10:]  # Last 10 alerts
            }
        
        return summary


class AlgorithmProfiler:
    """
    Detailed profiling system for recommendation algorithms.
    
    Provides line-by-line profiling and detailed performance analysis
    of algorithm implementations.
    """
    
    def __init__(self):
        self.profile_data = {}
        self.comparison_results = {}
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Profile a function with detailed statistics.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Profiling results
        """
        profiler = cProfile.Profile()
        
        start_time = time.perf_counter()
        
        # Run profiling
        profiler.enable()
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            result = None
        finally:
            profiler.disable()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Get profiling statistics
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        
        # Extract top functions
        top_functions = []
        for func_info, (call_count, total_time_func, cum_time, _, _) in stats.stats.items():
            filename, line_num, func_name = func_info
            top_functions.append({
                'function': func_name,
                'filename': os.path.basename(filename),
                'line': line_num,
                'calls': call_count,
                'total_time': total_time_func,
                'cumulative_time': cum_time,
                'per_call': total_time_func / call_count if call_count > 0 else 0
            })
        
        # Sort by cumulative time
        top_functions.sort(key=lambda x: x['cumulative_time'], reverse=True)
        
        profile_result = {
            'success': success,
            'error': error,
            'total_execution_time': total_time,
            'top_functions': top_functions[:20],  # Top 20 functions
            'total_function_calls': stats.total_calls,
            'primitive_calls': stats.prim_calls
        }
        
        return profile_result
    
    def compare_algorithms(self, algorithm_configs: List[Dict], 
                         test_data: Any) -> Dict[str, Any]:
        """
        Compare multiple algorithms with detailed profiling.
        
        Args:
            algorithm_configs: List of algorithm configurations
            test_data: Test data to run algorithms on
            
        Returns:
            Comparison results
        """
        comparison_results = {
            'algorithms': {},
            'rankings': {},
            'summary': {}
        }
        
        print("Profiling and comparing algorithms...")
        
        for config in algorithm_configs:
            algo_name = config['name']
            algo_func = config['func']
            
            print(f"  Profiling {algo_name}...")
            
            # Create wrapped function with parameters
            def wrapped_func():
                return algo_func(test_data, **config.get('params', {}))
            
            # Profile the algorithm
            profile_result = self.profile_function(wrapped_func)
            comparison_results['algorithms'][algo_name] = profile_result
        
        # Create rankings
        successful_algorithms = {
            name: results for name, results in comparison_results['algorithms'].items()
            if results['success']
        }
        
        if successful_algorithms:
            # Rank by execution time
            time_ranking = sorted(
                successful_algorithms.items(),
                key=lambda x: x[1]['total_execution_time']
            )
            comparison_results['rankings']['by_execution_time'] = [name for name, _ in time_ranking]
            
            # Rank by function calls
            calls_ranking = sorted(
                successful_algorithms.items(),
                key=lambda x: x[1]['total_function_calls']
            )
            comparison_results['rankings']['by_function_calls'] = [name for name, _ in calls_ranking]
        
        # Summary statistics
        if successful_algorithms:
            times = [results['total_execution_time'] for results in successful_algorithms.values()]
            calls = [results['total_function_calls'] for results in successful_algorithms.values()]
            
            if NUMPY_PANDAS_AVAILABLE:
                comparison_results['summary'] = {
                    'fastest_algorithm': comparison_results['rankings']['by_execution_time'][0],
                    'slowest_algorithm': comparison_results['rankings']['by_execution_time'][-1],
                    'avg_execution_time': np.mean(times),
                    'std_execution_time': np.std(times),
                    'avg_function_calls': np.mean(calls)
                }
        
        return comparison_results


class MemoryProfiler:
    """
    Memory usage profiling and leak detection.
    """
    
    def __init__(self):
        self.memory_snapshots = []
        self.allocation_tracking = defaultdict(list)
    
    @contextmanager
    def memory_monitor(self, operation_name: str):
        """
        Context manager for monitoring memory usage during operations.
        
        Args:
            operation_name: Name of the operation being monitored
        """
        # Force garbage collection before measurement
        gc.collect()
        
        # Get initial memory usage
        if SYSTEM_MONITORING_AVAILABLE:
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
        else:
            memory_before = 0.0
        
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            
            # Get final memory usage
            if SYSTEM_MONITORING_AVAILABLE:
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
            else:
                memory_after = 0.0
            
            memory_delta = memory_after - memory_before
            duration = end_time - start_time
            
            # Store memory usage data
            snapshot = {
                'operation': operation_name,
                'timestamp': start_time,
                'duration': duration,
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'memory_delta_mb': memory_delta,
                'memory_rate_mb_per_sec': memory_delta / duration if duration > 0 else 0
            }
            
            self.memory_snapshots.append(snapshot)
            self.allocation_tracking[operation_name].append(memory_delta)
    
    def detect_memory_leaks(self, operation_name: str, 
                          threshold_mb: float = 10.0) -> Dict[str, Any]:
        """
        Analyze memory allocation patterns to detect potential leaks.
        
        Args:
            operation_name: Operation to analyze
            threshold_mb: Memory leak threshold in MB
            
        Returns:
            Leak detection results
        """
        if operation_name not in self.allocation_tracking:
            return {'error': f'No data for operation: {operation_name}'}
        
        allocations = self.allocation_tracking[operation_name]
        
        if len(allocations) < 3:
            return {'error': 'Insufficient data points for analysis'}
        
        # Calculate trend
        if NUMPY_PANDAS_AVAILABLE:
            # Linear regression to find trend
            x = np.arange(len(allocations))
            coefficients = np.polyfit(x, allocations, 1)
            trend_slope = coefficients[0]
            
            # Statistics
            mean_allocation = np.mean(allocations)
            std_allocation = np.std(allocations)
            max_allocation = np.max(allocations)
            
        else:
            # Simple trend calculation without numpy
            trend_slope = (allocations[-1] - allocations[0]) / len(allocations)
            mean_allocation = sum(allocations) / len(allocations)
            max_allocation = max(allocations)
            std_allocation = 0.0
        
        # Leak detection logic
        potential_leak = (
            trend_slope > threshold_mb / len(allocations) or
            max_allocation > threshold_mb
        )
        
        leak_analysis = {
            'operation': operation_name,
            'total_measurements': len(allocations),
            'mean_allocation_mb': mean_allocation,
            'max_allocation_mb': max_allocation,
            'trend_slope_mb_per_iteration': trend_slope,
            'potential_leak_detected': potential_leak,
            'leak_severity': 'high' if max_allocation > threshold_mb * 2 else 'medium' if potential_leak else 'low'
        }
        
        if NUMPY_PANDAS_AVAILABLE:
            leak_analysis['std_allocation_mb'] = std_allocation
        
        return leak_analysis
    
    def generate_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report."""
        if not self.memory_snapshots:
            return {'error': 'No memory snapshots available'}
        
        report = {
            'total_snapshots': len(self.memory_snapshots),
            'operations': {},
            'overall_stats': {},
            'leak_analysis': {}
        }
        
        # Group by operation
        operations = defaultdict(list)
        for snapshot in self.memory_snapshots:
            operations[snapshot['operation']].append(snapshot)
        
        # Analyze each operation
        for op_name, snapshots in operations.items():
            memory_deltas = [s['memory_delta_mb'] for s in snapshots]
            durations = [s['duration'] for s in snapshots]
            
            if NUMPY_PANDAS_AVAILABLE and memory_deltas:
                op_stats = {
                    'count': len(snapshots),
                    'avg_memory_delta_mb': np.mean(memory_deltas),
                    'max_memory_delta_mb': np.max(memory_deltas),
                    'min_memory_delta_mb': np.min(memory_deltas),
                    'std_memory_delta_mb': np.std(memory_deltas),
                    'avg_duration_sec': np.mean(durations)
                }
            elif memory_deltas:
                op_stats = {
                    'count': len(snapshots),
                    'avg_memory_delta_mb': sum(memory_deltas) / len(memory_deltas),
                    'max_memory_delta_mb': max(memory_deltas),
                    'min_memory_delta_mb': min(memory_deltas),
                    'avg_duration_sec': sum(durations) / len(durations)
                }
            else:
                op_stats = {'count': 0}
            
            report['operations'][op_name] = op_stats
            
            # Leak analysis for this operation
            leak_analysis = self.detect_memory_leaks(op_name)
            if 'error' not in leak_analysis:
                report['leak_analysis'][op_name] = leak_analysis
        
        # Overall statistics
        all_deltas = [s['memory_delta_mb'] for s in self.memory_snapshots]
        if all_deltas and NUMPY_PANDAS_AVAILABLE:
            report['overall_stats'] = {
                'total_memory_allocated_mb': np.sum([d for d in all_deltas if d > 0]),
                'total_memory_freed_mb': abs(np.sum([d for d in all_deltas if d < 0])),
                'net_memory_change_mb': np.sum(all_deltas),
                'avg_memory_delta_mb': np.mean(all_deltas)
            }
        
        return report


def performance_decorator(monitor: PerformanceMonitor, operation_name: str):
    """
    Decorator for automatic performance monitoring of functions.
    
    Args:
        monitor: PerformanceMonitor instance
        operation_name: Name of the operation for tracking
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with monitor.operation_timer(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class PerformanceReporter:
    """
    Generate comprehensive performance reports and visualizations.
    """
    
    def __init__(self):
        self.reports = []
    
    def generate_comprehensive_report(self, monitor: PerformanceMonitor,
                                    profiler: AlgorithmProfiler,
                                    memory_profiler: MemoryProfiler) -> Dict[str, Any]:
        """
        Generate a comprehensive performance analysis report.
        
        Args:
            monitor: Performance monitor instance
            profiler: Algorithm profiler instance
            memory_profiler: Memory profiler instance
            
        Returns:
            Comprehensive performance report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_performance': {},
            'algorithm_profiling': {},
            'memory_analysis': {},
            'recommendations': []
        }
        
        # System performance summary
        try:
            system_summary = monitor.get_performance_summary(time_window=3600)
            report['system_performance'] = system_summary
        except Exception as e:
            report['system_performance'] = {'error': str(e)}
        
        # Memory analysis
        try:
            memory_report = memory_profiler.generate_memory_report()
            report['memory_analysis'] = memory_report
        except Exception as e:
            report['memory_analysis'] = {'error': str(e)}
        
        # Generate recommendations based on analysis
        recommendations = self._generate_recommendations(report)
        report['recommendations'] = recommendations
        
        # Store report
        self.reports.append(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze system performance
        sys_perf = report.get('system_performance', {})
        metrics_summary = sys_perf.get('metrics_summary', {})
        
        # CPU recommendations
        cpu_data = metrics_summary.get('cpu_percent', {})
        if cpu_data and cpu_data.get('mean', 0) > 70:
            recommendations.append(
                "High CPU usage detected. Consider optimizing algorithms or using "
                "more efficient data structures."
            )
        
        # Memory recommendations
        memory_data = metrics_summary.get('memory_percent', {})
        if memory_data and memory_data.get('mean', 0) > 80:
            recommendations.append(
                "High memory usage detected. Consider implementing data streaming "
                "or using more memory-efficient algorithms."
            )
        
        # Memory leak recommendations
        memory_analysis = report.get('memory_analysis', {})
        leak_analysis = memory_analysis.get('leak_analysis', {})
        
        for operation, leak_info in leak_analysis.items():
            if leak_info.get('potential_leak_detected', False):
                recommendations.append(
                    f"Potential memory leak detected in {operation}. "
                    f"Review object cleanup and garbage collection."
                )
        
        # Operation performance recommendations
        op_summary = sys_perf.get('operation_summary', {})
        for op_name, op_data in op_summary.items():
            error_rate = op_data.get('error_rate', 0)
            if error_rate > 0.01:  # 1% error rate
                recommendations.append(
                    f"High error rate ({error_rate:.2%}) in {op_name}. "
                    f"Review error handling and input validation."
                )
            
            avg_time = op_data.get('avg_time', 0)
            if avg_time > 1.0:  # 1 second threshold
                recommendations.append(
                    f"Slow operation detected: {op_name} (avg: {avg_time:.2f}s). "
                    f"Consider optimization or caching."
                )
        
        return recommendations
    
    def export_report(self, report: Dict[str, Any], filename: str) -> None:
        """Export report to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Report exported to {filename}")
        except Exception as e:
            print(f"Error exporting report: {e}")
    
    def create_performance_dashboard(self, monitor: PerformanceMonitor,
                                   output_file: str = "performance_dashboard.png") -> None:
        """
        Create a visual performance dashboard.
        
        Args:
            monitor: Performance monitor with data
            output_file: Output file for the dashboard
        """
        if not VISUALIZATION_AVAILABLE:
            print("Visualization libraries not available. Skipping dashboard creation.")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Recommendation System Performance Dashboard', fontsize=16)
            
            # Get performance data
            with monitor.lock:
                # CPU usage over time
                cpu_data = monitor.performance_data.get('cpu_percent', [])
                if cpu_data:
                    times, values = zip(*cpu_data)
                    axes[0, 0].plot(times, values)
                    axes[0, 0].set_title('CPU Usage Over Time')
                    axes[0, 0].set_ylabel('CPU %')
                
                # Memory usage over time
                memory_data = monitor.performance_data.get('memory_percent', [])
                if memory_data:
                    times, values = zip(*memory_data)
                    axes[0, 1].plot(times, values, color='orange')
                    axes[0, 1].set_title('Memory Usage Over Time')
                    axes[0, 1].set_ylabel('Memory %')
                
                # Operation performance
                operation_names = list(monitor.operation_times.keys())
                if operation_names:
                    avg_times = [sum(monitor.operation_times[op]) / len(monitor.operation_times[op])
                               for op in operation_names]
                    axes[1, 0].bar(operation_names, avg_times)
                    axes[1, 0].set_title('Average Operation Times')
                    axes[1, 0].set_ylabel('Time (seconds)')
                    axes[1, 0].tick_params(axis='x', rotation=45)
                
                # Error rates
                if operation_names:
                    error_rates = [monitor.error_counts[op] / monitor.operation_counts[op] * 100
                                 if monitor.operation_counts[op] > 0 else 0
                                 for op in operation_names]
                    axes[1, 1].bar(operation_names, error_rates, color='red', alpha=0.7)
                    axes[1, 1].set_title('Operation Error Rates')
                    axes[1, 1].set_ylabel('Error Rate (%)')
                    axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Performance dashboard saved to {output_file}")
            
        except Exception as e:
            print(f"Error creating performance dashboard: {e}")
    
    def print_performance_summary(self, report: Dict[str, Any]) -> None:
        """Print a human-readable performance summary."""
        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("="*80)
        
        # System Performance
        sys_perf = report.get('system_performance', {})
        metrics = sys_perf.get('metrics_summary', {})
        
        if metrics:
            print("\nSYSTEM PERFORMANCE:")
            print("-" * 40)
            
            cpu_data = metrics.get('cpu_percent', {})
            if cpu_data:
                print(f"CPU Usage: Avg {cpu_data.get('mean', 0):.1f}%, "
                      f"Max {cpu_data.get('max', 0):.1f}%")
            
            memory_data = metrics.get('memory_percent', {})
            if memory_data:
                print(f"Memory Usage: Avg {memory_data.get('mean', 0):.1f}%, "
                      f"Max {memory_data.get('max', 0):.1f}%")
        
        # Operation Performance
        op_summary = sys_perf.get('operation_summary', {})
        if op_summary:
            print("\nOPERATION PERFORMANCE:")
            print("-" * 40)
            
            for op_name, op_data in op_summary.items():
                print(f"{op_name}:")
                print(f"  Operations: {op_data.get('total_operations', 0)}")
                print(f"  Avg Time: {op_data.get('avg_time', 0):.4f}s")
                print(f"  Error Rate: {op_data.get('error_count', 0) / op_data.get('total_operations', 1) * 100:.2f}%")
        
        # Memory Analysis
        memory_analysis = report.get('memory_analysis', {})
        leak_analysis = memory_analysis.get('leak_analysis', {})
        
        if leak_analysis:
            print("\nMEMORY LEAK ANALYSIS:")
            print("-" * 40)
            
            for operation, leak_info in leak_analysis.items():
                status = "⚠️ POTENTIAL LEAK" if leak_info.get('potential_leak_detected') else "✅ OK"
                print(f"{operation}: {status}")
                if leak_info.get('potential_leak_detected'):
                    print(f"  Max allocation: {leak_info.get('max_allocation_mb', 0):.2f} MB")
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            print("\nRECOMMENDATIONS:")
            print("-" * 40)
            
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        
        print("\n" + "="*80)