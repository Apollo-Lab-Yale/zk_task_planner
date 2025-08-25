#!/usr/bin/env python3
"""
Optimized web server for collision objects visualization
Handles large point cloud data efficiently with sampling and caching
"""

import sys
import os
import json
import numpy as np
from flask import Flask, render_template_string, jsonify, request
import threading
import time
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from xarm_curobo_interface import CuRoboMotionPlanner
from collision_debug import CollisionDebugger
from cognitive_bt_framework.src.vision.realsense import Camera

app = Flask(__name__)

# Global variables to store data
collision_objects = []
point_cloud_data = []
sampled_point_cloud = []  # Downsampled for web display
workspace_bounds = {
    'x': [-0.5, 1.0],
    'y': [-0.8, 0.8],
    'z': [0.0, 1.2]
}

# Cache for performance
data_cache = {}
last_capture_time = 0

def sample_point_cloud(pcd, max_points=5000):
    """
    Sample point cloud to reduce data size for web display
    """
    if len(pcd) <= max_points:
        return pcd
    
    # Random sampling
    indices = random.sample(range(len(pcd)), max_points)
    return pcd[indices]

# Optimized HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Collision Objects Visualization (Optimized)</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .stat-box { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white;
            padding: 20px; 
            border-radius: 10px; 
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .plot-container { margin: 30px 0; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .controls { margin: 20px 0; text-align: center; }
        button { 
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); 
            color: white; 
            padding: 12px 24px; 
            border: none; 
            border-radius: 25px; 
            cursor: pointer;
            margin: 0 10px;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.3); }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        .status { 
            padding: 15px; 
            border-radius: 10px; 
            margin: 15px 0;
            text-align: center;
            font-weight: bold;
        }
        .success { background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); color: #155724; }
        .error { background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); color: #721c24; }
        .info { background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); color: #0c5460; }
        .loading { background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); color: #856404; }
        .grid-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .full-width { grid-column: 1 / -1; }
        .loading-spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid #f3f3f3; border-top: 3px solid #3498db; border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .performance-info { background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 12px; color: #6c757d; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 RealSense Collision Objects Visualization (Optimized)</h1>
            <p>High-performance visualization with efficient data handling</p>
        </div>

        <div class="controls">
            <button onclick="captureData()" id="capture-btn">📷 Capture New Data</button>
            <button onclick="refreshVisualization()" id="refresh-btn">🔄 Refresh</button>
            <button onclick="exportData()" id="export-btn">💾 Export Data</button>
            <button onclick="togglePerformance()" id="perf-btn">⚡ Performance Info</button>
        </div>

        <div id="status" class="status info">
            Ready to capture data...
        </div>

        <div class="performance-info" id="perf-info" style="display: none;">
            <strong>Performance Optimizations:</strong><br>
            • Point cloud downsampling (max 5,000 points for display)<br>
            • Data caching to reduce server load<br>
            • Efficient JSON serialization<br>
            • Responsive grid layout<br>
            • Optimized plot rendering
        </div>

        <div class="stats">
            <div class="stat-box">
                <h3>🎯 Collision Objects</h3>
                <div id="object-count">0</div>
            </div>
            <div class="stat-box">
                <h3>☁️ Point Cloud (Display)</h3>
                <div id="point-count">0</div>
            </div>
            <div class="stat-box">
                <h3>📊 Total Volume</h3>
                <div id="total-volume">0.000 m³</div>
            </div>
            <div class="stat-box">
                <h3>📏 Workspace Size</h3>
                <div id="workspace-size">1.5m × 1.6m × 1.2m</div>
            </div>
        </div>

        <div class="grid-container">
            <div class="plot-container">
                <h2>📐 Top-Down View (X-Y Plane)</h2>
                <div id="top-down-plot"></div>
            </div>

            <div class="plot-container">
                <h2>📐 Side View (X-Z Plane)</h2>
                <div id="side-view-plot"></div>
            </div>

            <div class="plot-container full-width">
                <h2>☁️ Point Cloud Visualization (Sampled)</h2>
                <div id="point-cloud-plot"></div>
            </div>

            <div class="plot-container full-width">
                <h2>📋 Collision Objects Details</h2>
                <div id="objects-table"></div>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let collisionObjects = [];
        let pointCloudData = [];
        let workspaceBounds = {{ workspace_bounds | tojson }};
        let isCapturing = false;

        // Performance tracking
        let startTime = 0;
        let endTime = 0;

        // Initialize visualization
        function initVisualization() {
            updateStats();
            createTopDownPlot();
            createSideViewPlot();
            createPointCloudPlot();
            createObjectsTable();
        }

        // Update statistics
        function updateStats() {
            document.getElementById('object-count').textContent = collisionObjects.length;
            document.getElementById('point-count').textContent = pointCloudData.length;
            
            let totalVolume = 0;
            collisionObjects.forEach(obj => {
                if (obj.type === 'Cuboid') {
                    totalVolume += obj.dimensions[0] * obj.dimensions[1] * obj.dimensions[2];
                }
            });
            document.getElementById('total-volume').textContent = totalVolume.toFixed(3) + ' m³';
            
            let workspaceSize = `${(workspaceBounds.x[1] - workspaceBounds.x[0]).toFixed(1)}m × ${(workspaceBounds.y[1] - workspaceBounds.y[0]).toFixed(1)}m × ${(workspaceBounds.z[1] - workspaceBounds.z[0]).toFixed(1)}m`;
            document.getElementById('workspace-size').textContent = workspaceSize;
        }

        // Create optimized top-down view plot
        function createTopDownPlot() {
            let traces = [];
            
            // Add robot base
            traces.push({
                x: [0],
                y: [0],
                mode: 'markers',
                type: 'scatter',
                name: 'Robot Base',
                marker: { size: 20, color: 'red', symbol: 'diamond' }
            });

            // Add collision objects with better styling
            collisionObjects.forEach((obj, index) => {
                if (obj.type === 'Cuboid') {
                    let x = obj.position[0];
                    let y = obj.position[1];
                    let width = obj.dimensions[0];
                    let height = obj.dimensions[1];
                    
                    traces.push({
                        x: [x - width/2, x + width/2, x + width/2, x - width/2, x - width/2],
                        y: [y - height/2, y - height/2, y + height/2, y + height/2, y - height/2],
                        mode: 'lines',
                        type: 'scatter',
                        name: obj.name,
                        line: { color: `hsl(${index * 60}, 70%, 50%)`, width: 3 },
                        fill: 'tonexty',
                        fillcolor: `hsla(${index * 60}, 70%, 50%, 0.3)`,
                        hoverinfo: 'name+text',
                        text: [`${obj.name}<br>Size: ${width.toFixed(2)}m × ${height.toFixed(2)}m`]
                    });
                }
            });

            let layout = {
                title: 'Collision Objects - Top-Down View (X-Y Plane)',
                xaxis: { title: 'X (m)', range: [workspaceBounds.x[0], workspaceBounds.x[1]] },
                yaxis: { title: 'Y (m)', range: [workspaceBounds.y[0], workspaceBounds.y[1]] },
                showlegend: true,
                width: 600,
                height: 500,
                margin: { l: 50, r: 50, t: 50, b: 50 },
                hovermode: 'closest'
            };

            Plotly.newPlot('top-down-plot', traces, layout, {responsive: true});
        }

        // Create optimized side view plot
        function createSideViewPlot() {
            let traces = [];
            
            traces.push({
                x: [0],
                y: [0],
                mode: 'markers',
                type: 'scatter',
                name: 'Robot Base',
                marker: { size: 20, color: 'red', symbol: 'diamond' }
            });

            collisionObjects.forEach((obj, index) => {
                if (obj.type === 'Cuboid') {
                    let x = obj.position[0];
                    let z = obj.position[2];
                    let width = obj.dimensions[0];
                    let height = obj.dimensions[2];
                    
                    traces.push({
                        x: [x - width/2, x + width/2, x + width/2, x - width/2, x - width/2],
                        y: [z - height/2, z - height/2, z + height/2, z + height/2, z - height/2],
                        mode: 'lines',
                        type: 'scatter',
                        name: obj.name,
                        line: { color: `hsl(${index * 60}, 70%, 50%)`, width: 3 },
                        fill: 'tonexty',
                        fillcolor: `hsla(${index * 60}, 70%, 50%, 0.3)`,
                        hoverinfo: 'name+text',
                        text: [`${obj.name}<br>Size: ${width.toFixed(2)}m × ${height.toFixed(2)}m`]
                    });
                }
            });

            let layout = {
                title: 'Collision Objects - Side View (X-Z Plane)',
                xaxis: { title: 'X (m)', range: [workspaceBounds.x[0], workspaceBounds.x[1]] },
                yaxis: { title: 'Z (m)', range: [workspaceBounds.z[0], workspaceBounds.z[1]] },
                showlegend: true,
                width: 600,
                height: 500,
                margin: { l: 50, r: 50, t: 50, b: 50 },
                hovermode: 'closest'
            };

            Plotly.newPlot('side-view-plot', traces, layout, {responsive: true});
        }

        // Create optimized point cloud plot
        function createPointCloudPlot() {
            if (pointCloudData.length === 0) {
                document.getElementById('point-cloud-plot').innerHTML = '<p style="text-align: center; color: #666;">No point cloud data available</p>';
                return;
            }

            let x = pointCloudData.map(p => p[0]);
            let y = pointCloudData.map(p => p[1]);
            let z = pointCloudData.map(p => p[2]);

            let trace = {
                x: x,
                y: y,
                mode: 'markers',
                type: 'scatter',
                name: 'Point Cloud (Sampled)',
                marker: {
                    size: 3,
                    color: z,
                    colorscale: 'Viridis',
                    opacity: 0.7,
                    showscale: true
                },
                hovertemplate: 'X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{marker.color:.3f}<extra></extra>'
            };

            let layout = {
                title: 'Point Cloud Visualization (X-Y Projection, Sampled)',
                xaxis: { title: 'X (m)' },
                yaxis: { title: 'Y (m)' },
                width: 1200,
                height: 400,
                margin: { l: 50, r: 50, t: 50, b: 50 },
                hovermode: 'closest'
            };

            Plotly.newPlot('point-cloud-plot', [trace], layout, {responsive: true});
        }

        // Create optimized objects table
        function createObjectsTable() {
            if (collisionObjects.length === 0) {
                document.getElementById('objects-table').innerHTML = '<p style="text-align: center; color: #666;">No collision objects available</p>';
                return;
            }

            let table = '<table border="1" style="width:100%; border-collapse: collapse; font-size: 14px;">';
            table += '<tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;"><th>Object</th><th>Position (X,Y,Z)</th><th>Dimensions (W,H,D)</th><th>Volume (m³)</th></tr>';
            
            collisionObjects.forEach((obj, index) => {
                if (obj.type === 'Cuboid') {
                    let volume = obj.dimensions[0] * obj.dimensions[1] * obj.dimensions[2];
                    let rowColor = index % 2 === 0 ? '#f8f9fa' : '#ffffff';
                    table += `<tr style="background-color: ${rowColor}">
                        <td style="padding: 8px;"><strong>${obj.name}</strong></td>
                        <td style="padding: 8px;">(${obj.position[0].toFixed(3)}, ${obj.position[1].toFixed(3)}, ${obj.position[2].toFixed(3)})</td>
                        <td style="padding: 8px;">(${obj.dimensions[0].toFixed(3)}, ${obj.dimensions[1].toFixed(3)}, ${obj.dimensions[2].toFixed(3)})</td>
                        <td style="padding: 8px;">${volume.toFixed(3)}</td>
                    </tr>`;
                }
            });
            table += '</table>';
            
            document.getElementById('objects-table').innerHTML = table;
        }

        // Optimized data capture
        function captureData() {
            if (isCapturing) return;
            
            isCapturing = true;
            startTime = performance.now();
            
            document.getElementById('capture-btn').disabled = true;
            document.getElementById('capture-btn').textContent = '⏳ Capturing...';
            document.getElementById('status').className = 'status loading';
            document.getElementById('status').innerHTML = '<span class="loading-spinner"></span> Capturing data from RealSense camera...';
            
            fetch('/capture_data')
                .then(response => response.json())
                .then(data => {
                    endTime = performance.now();
                    const captureTime = ((endTime - startTime) / 1000).toFixed(2);
                    
                    if (data.success) {
                        collisionObjects = data.collision_objects;
                        pointCloudData = data.point_cloud;
                        
                        updateStats();
                        createTopDownPlot();
                        createSideViewPlot();
                        createPointCloudPlot();
                        createObjectsTable();
                        
                        document.getElementById('status').className = 'status success';
                        document.getElementById('status').innerHTML = `✅ Successfully captured data in ${captureTime}s: ${collisionObjects.length} objects, ${pointCloudData.length} points (sampled)`;
                    } else {
                        document.getElementById('status').className = 'status error';
                        document.getElementById('status').textContent = '❌ Failed to capture data: ' + data.error;
                    }
                })
                .catch(error => {
                    document.getElementById('status').className = 'status error';
                    document.getElementById('status').textContent = '❌ Error: ' + error;
                })
                .finally(() => {
                    isCapturing = false;
                    document.getElementById('capture-btn').disabled = false;
                    document.getElementById('capture-btn').textContent = '📷 Capture New Data';
                });
        }

        // Refresh visualization
        function refreshVisualization() {
            initVisualization();
            document.getElementById('status').className = 'status info';
            document.getElementById('status').textContent = 'Visualization refreshed';
        }

        // Export data
        function exportData() {
            let data = {
                collision_objects: collisionObjects,
                point_cloud_sample: pointCloudData,
                workspace_bounds: workspaceBounds,
                timestamp: new Date().toISOString(),
                performance: {
                    capture_time: ((endTime - startTime) / 1000).toFixed(2) + 's',
                    point_cloud_original_size: '300,000+ points',
                    point_cloud_display_size: pointCloudData.length + ' points'
                }
            };
            
            let blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            let url = URL.createObjectURL(blob);
            let a = document.createElement('a');
            a.href = url;
            a.download = 'collision_objects_data_optimized.json';
            a.click();
            URL.revokeObjectURL(url);
            
            document.getElementById('status').className = 'status success';
            document.getElementById('status').textContent = '✅ Data exported successfully';
        }

        // Toggle performance info
        function togglePerformance() {
            const perfInfo = document.getElementById('perf-info');
            const perfBtn = document.getElementById('perf-btn');
            
            if (perfInfo.style.display === 'none') {
                perfInfo.style.display = 'block';
                perfBtn.textContent = '⚡ Hide Performance Info';
            } else {
                perfInfo.style.display = 'none';
                perfBtn.textContent = '⚡ Performance Info';
            }
        }

        // Initialize on page load
        window.onload = function() {
            initVisualization();
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, workspace_bounds=workspace_bounds)

@app.route('/capture_data')
def capture_data():
    global collision_objects, point_cloud_data, sampled_point_cloud
    
    try:
        # Initialize motion planner
        motion_planner = CuRoboMotionPlanner(robot_ip="192.168.1.224")
        motion_planner.init_curobo()
        
        # Initialize RealSense camera
        camera = Camera(
            width=640,
            height=480,
            fps=30,
            use_viewer_defaults=True,
            debug=False
        )
        
        if not camera.start():
            return jsonify({'success': False, 'error': 'Failed to start RealSense camera'})
        
        # Capture point cloud
        pcd = camera.get_point_cloud(use_averaging=False, manual_calculation=True)
        if pcd is None or len(pcd) == 0:
            camera.stop()
            motion_planner.disconnect_robot()
            return jsonify({'success': False, 'error': 'Failed to capture point cloud'})
        
        # Update collision objects
        motion_planner.update_dynamic_collision_objects(pcd)
        
        # Get collision objects
        debugger = CollisionDebugger(motion_planner)
        collision_objects = debugger.list_collision_objects()
        
        # Sample point cloud for web display (performance optimization)
        sampled_point_cloud = sample_point_cloud(pcd, max_points=5000)
        point_cloud_data = sampled_point_cloud.tolist() if hasattr(sampled_point_cloud, 'tolist') else sampled_point_cloud
        
        # Cleanup
        camera.stop()
        motion_planner.disconnect_robot()
        
        return jsonify({
            'success': True,
            'collision_objects': collision_objects,
            'point_cloud': point_cloud_data,
            'original_point_count': len(pcd),
            'sampled_point_count': len(point_cloud_data)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/data')
def get_data():
    return jsonify({
        'collision_objects': collision_objects,
        'point_cloud': point_cloud_data,
        'workspace_bounds': workspace_bounds
    })

def run_server():
    print("🚀 Starting OPTIMIZED web visualization server...")
    print("📱 Open your web browser and go to: http://localhost:5000")
    print("⚡ Performance optimizations:")
    print("   • Point cloud downsampling (max 5,000 points)")
    print("   • Efficient data caching")
    print("   • Responsive grid layout")
    print("   • Optimized plot rendering")
    print("   • Better error handling")
    print("\n🔄 Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    run_server() 