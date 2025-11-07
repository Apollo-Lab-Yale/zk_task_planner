import numpy as np
import time
import functools
from collections import defaultdict

class IterationTimeProfiler:
    """Time profiler that focuses on tracking performance per iteration"""
    
    def __init__(self, enabled=False):
        self.enabled = enabled
        # Store historical data for statistics
        self.historical_timings = defaultdict(list)
        # Store current iteration timings
        self.current_timings = defaultdict(list)
        # Track active timing sections
        self.start_times = {}
        self.active_sections = set()
        # Track section hierarchy
        self.section_hierarchy = {}
        self.section_depth = {}
        # Track iteration count
        self.iteration_count = 0
        # Track if we're in an iteration
        self.in_iteration = False
        
    def start_iteration(self):
        """Start a new timing iteration, resetting current timings"""
        if not self.enabled:
            return
            
        # Clear current timing data but keep hierarchy info
        self.current_timings.clear()
        self.start_times.clear()
        self.active_sections.clear()
        self.in_iteration = True
        self.iteration_count += 1
        
        if self.iteration_count == 1:
            print("First iteration started - timing data collection begins")
        
    def end_iteration(self, preserve_current=False):
        """
        End the current iteration and update historical stats
        
        Args:
            preserve_current: If True, keeps current timing data for reporting
        """
        if not self.enabled or not self.in_iteration:
            return
            
        # Make sure all sections are stopped
        active_sections_copy = list(self.active_sections)
        for section in active_sections_copy:
            print(f"Warning: Section '{section}' was not stopped before end_iteration")
            self.stop(section)
        
        # Save current timings before updating historical data
        current_timings_copy = None
        if preserve_current:
            current_timings_copy = {k: list(v) for k, v in self.current_timings.items()}
            
        # Update historical timings with current iteration data
        for section, times in self.current_timings.items():
            if times:
                # Store total time for this section in this iteration
                total_time = sum(times)
                self.historical_timings[section].append(total_time)
        
        # Restore current timings if requested
        if preserve_current:
            self.current_timings = defaultdict(list, current_timings_copy)
        else:
            self.current_timings.clear()
                
        self.in_iteration = False
        
    def start(self, section_name):
        """Start timing a section"""
        if not self.enabled:
            return
        
        # If already timing this section, stop it first to avoid nesting errors
        if section_name in self.active_sections:
            self.stop(section_name)
            
        self.start_times[section_name] = time.time()
        self.active_sections.add(section_name)
        
        # Track hierarchy for nested sections
        parent = None
        for section in self.active_sections:
            if section != section_name and section in self.start_times:
                if parent is None or self.section_depth.get(section, 0) > self.section_depth.get(parent, 0):
                    parent = section
                    
        if parent:
            self.section_hierarchy[section_name] = parent
            self.section_depth[section_name] = self.section_depth.get(parent, 0) + 1
        else:
            self.section_depth[section_name] = 0
        
    def stop(self, section_name):
        """Stop timing a section and record the elapsed time"""
        if not self.enabled or section_name not in self.start_times:
            return None
            
        elapsed = time.time() - self.start_times[section_name]
        
        # Store timing regardless of iteration status
        self.current_timings[section_name].append(elapsed)
            
        del self.start_times[section_name]
        if section_name in self.active_sections:
            self.active_sections.remove(section_name)
        return elapsed
    
    def get_current_iteration_total(self, section_name):
        """Get total time for a section in the current iteration"""
        if not self.enabled:
            return 0
        times = self.current_timings.get(section_name, [])
        return sum(times) if times else 0
    
    def get_avg_time(self, section_name):
        """Get average time across all iterations for a section"""
        if not self.enabled:
            return 0
        times = self.historical_timings.get(section_name, [])
        return np.mean(times) if times else 0
    
    def get_min_time(self, section_name):
        """Get minimum time across all iterations for a section"""
        if not self.enabled:
            return 0
        times = self.historical_timings.get(section_name, [])
        return np.min(times) if times else 0
    
    def get_max_time(self, section_name):
        """Get maximum time across all iterations for a section"""
        if not self.enabled:
            return 0
        times = self.historical_timings.get(section_name, [])
        return np.max(times) if times else 0
        
    def reset(self):
        """Reset all timing data"""
        self.historical_timings.clear()
        self.current_timings.clear()
        self.start_times.clear()
        self.active_sections.clear()
        self.section_hierarchy.clear()
        self.section_depth.clear()
        self.iteration_count = 0
        self.in_iteration = False
    
    def get_summary(self, sort_by="total", include_empty_sections=False, top_n=None):
        """
        Get summary of timing information
        
        Args:
            sort_by: Field to sort by (total, average, current_total, hierarchy)
            include_empty_sections: Whether to include sections with no timing data
            top_n: Limit to top N sections (by sort criteria)
            
        Returns:
            List of dictionaries with timing information
        """
        if not self.enabled:
            return []
            
        summary = []
        # Process all sections that have either current or historical data
        all_sections = set(list(self.current_timings.keys()) + list(self.historical_timings.keys()))
        
        for section in all_sections:
            # Current iteration data
            current_times = self.current_timings.get(section, [])
            current_total = sum(current_times) if current_times else 0
            current_avg = np.mean(current_times) if current_times else 0
            
            # Historical data
            historical_times = self.historical_timings.get(section, [])
            iterations_with_data = len(historical_times)
            
            # Skip if no data at all and we're not including empty sections
            if not include_empty_sections and not current_times and not historical_times:
                continue
                
            parent = self.section_hierarchy.get(section, None)
            depth = self.section_depth.get(section, 0)
            
            summary.append({
                'section': section,
                'depth': depth,
                'parent': parent,
                # Current iteration stats
                'current_total': current_total,
                'current_avg': current_avg,
                'current_calls': len(current_times),
                # Historical stats
                'iterations': iterations_with_data,
                'total': sum(historical_times) if historical_times else 0,
                'average': np.mean(historical_times) if historical_times else 0,
                'min': np.min(historical_times) if historical_times else 0,
                'max': np.max(historical_times) if historical_times else 0
            })
        
        # Sort by the specified key
        if sort_by == "hierarchy":
            summary.sort(key=lambda x: (x['depth'], x['total']), reverse=True)
        else:
            summary.sort(key=lambda x: x[sort_by], reverse=True)
        
        # Limit to top N if specified
        if top_n is not None and isinstance(top_n, int) and top_n > 0:
            summary = summary[:top_n]
            
        return summary
    
    def print_summary(self, sort_by="total", top_n=None, compact=False):
        """
        Print summary of timing information with improved formatting
        
        Args:
            sort_by: Field to sort by (total, average, current_total, hierarchy)
            top_n: Limit to top N sections (by sort criteria)
            compact: Use compact display format (less columns)
        """
        if not self.enabled:
            print("Time profiling is disabled.")
            return
            
        # Get summary without ending the iteration (preserves current data)
        summary = self.get_summary(sort_by, top_n=top_n)
        if not summary:
            print("No timing data collected")
            return
            
        # Get current iteration number and status
        is_active = "Active" if self.in_iteration else "Completed"
        current_iter = f"Iteration {self.iteration_count} ({is_active})"
        
        print(f"\n===== PERCEPTION SYSTEM TIMING SUMMARY ({current_iter}) =====")
        
        # Calculate the total overall time for root sections (depth=0)
        current_root_total = sum(item['current_total'] for item in summary if item['depth'] == 0)
        historical_avg_total = sum(item['average'] for item in summary if item['depth'] == 0 and item['iterations'] > 0)
        
        # Print header
        if compact:
            # Compact display with fewer columns
            print(f"{'Section':<40} {'Current(s)':<10} {'Avg(s)':<10} {'Calls':<8} {'% of Total':<10}")
            print("-" * 80)
        else:
            # Full display
            print(f"{'Section':<40} {'Current(s)':<10} {'Avg(s)':<10} {'Min(s)':<10} {'Max(s)':<10} {'Calls':<8} {'% of Total':<10}")
            print("-" * 100)
        
        # Create sorted and formatted section list
        formatted_sections = []
        for item in summary:
            # Indent name based on hierarchy
            section = item['section']
            depth = item['depth']
            indented_name = "  " * depth + section
            
            # Calculate percentage of relevant total
            if depth == 0:
                percentage = (item['current_total'] / current_root_total * 100) if current_root_total > 0 else 0
            else:
                parent = item['parent']
                parent_total = next((i['current_total'] for i in summary if i['section'] == parent), 0)
                percentage = (item['current_total'] / parent_total * 100) if parent_total > 0 else 0
            
            # Format the line based on compact mode
            if compact:
                line = f"{indented_name:<40} {item['current_total']:<10.3f} {item['average']:<10.3f} " \
                    f"{item['current_calls']:<8} {percentage:<10.1f}%"
            else:
                line = f"{indented_name:<40} {item['current_total']:<10.3f} {item['average']:<10.3f} " \
                    f"{item['min']:<10.3f} {item['max']:<10.3f} {item['current_calls']:<8} {percentage:<10.1f}%"
            
            formatted_sections.append((depth, line))
        
        # Sort by hierarchy first if requested
        if sort_by == "hierarchy":
            formatted_sections.sort(key=lambda x: x[0])
            
        # Print all sections with proper spacing
        for _, line in formatted_sections:
            print(line)
                
        print(f"\nCurrent iteration total: {current_root_total:.3f}s")
        if self.iteration_count > 0:
            iteration_count = max(1, self.iteration_count - (1 if self.in_iteration else 0))
            print(f"Average iteration total: {historical_avg_total:.3f}s (over {iteration_count} iterations)")
        print("=================================================")
        
    def timed(self, section_name):
        """Decorator to time a function or method"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                    
                self.start(section_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.stop(section_name)
            return wrapper
        return decorator
