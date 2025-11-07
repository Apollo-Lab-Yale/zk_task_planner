#!/usr/bin/env python3
"""
Web server for collision objects visualization
Provides a web interface to view collision objects and point cloud data
"""

import sys
import os
import json
import numpy as np
from flask import Flask, render_template_string, jsonify, request
import threading
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from xarm_curobo_interface import CuRoboMotionPlanner
from collision_debug import CollisionDebugger
from cognitive_bt_framework.src.vision.realsense import Camera

app = Flask(__name__)

# Global variables to store data
collision_objects = []
point_cloud_data = []
workspace_bounds = {
    'x': [-0.5, 1.0],
    'y': [-0.8, 0.8],
    'z': [0.0, 1.2]
}

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Collision Objects Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .stats { display: flex; justify-content: space-around; margin: 20px 0; }
        .stat-box { 
            background: #f0f0f0; 
            padding: 15px; 
            border-radius: 5px; 
            text-align: center;
            flex: 1;
            margin: 0 10px;
        }
        .plot-container { margin: 20px 0; }
        .controls { margin: 20px 0; text-align: center; }
        button { 
            background: #4CAF50; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            margin: 0 5px;
        }
        button:hover { background: #45a049; }
        .status { 
            padding: 10px; 
            border-radius: 5px; 
            margin: 10px 0;
            text-align: center;
        }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .info { background: #d1ecf1; color: #0c5460; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 RealSense Collision Objects Visualization</h1>
            <p>Live visualization of collision objects generated from RealSense camera data</p>
        </div>

        <div class="controls">
            <button onclick="captureData()">📷 Capture New Data</button>
            <button onclick="refreshVisualization()">🔄 Refresh Visualization</button>
            <button onclick="exportData()">💾 Export Data</button>
        </div>

        <div id="status" class="status info">
            Ready to capture data...
        </div>

        <div class="stats">
            <div class="stat-box">
                <h3>Collision Objects</h3>
                <div id="object-count">0</div>
            </div>
            <div class="stat-box">
                <h3>Point Cloud Points</h3>
                <div id="point-count">0</div>
            </div>
            <div class="stat-box">
                <h3>Total Volume</h3>
                <div id="total-volume">0.000 m³</div>
            </div>
            <div class="stat-box">
                <h3>Workspace Size</h3>
                <div id="workspace-size">1.5m × 1.6m × 1.2m</div>
            </div>
        </div>

        <div class="plot-container">
            <h2>Top-Down View (X-Y Plane)</h2>
            <div id="top-down-plot"></div>
        </div>

        <div class="plot-container">
            <h2>Side View (X-Z Plane)</h2>
            <div id="side-view-plot"></div>
        </div>

        <div class="plot-container">
            <h2>Point Cloud Visualization</h2>
            <div id="point-cloud-plot"></div>
        </div>

        <div class="plot-container">
            <h2>Collision Objects Details</h2>
            <div id="objects-table"></div>
        </div>
    </div>

    <script>
        // Global variables
        let collisionObjects = [];
        let pointCloudData = [];
        let workspaceBounds = {{ workspace_bounds | tojson }};

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

        // Create top-down view plot
        function createTopDownPlot() {
            let traces = [];
            
            // Add robot base
            traces.push({
                x: [0],
                y: [0],
                mode: 'markers',
                type: 'scatter',
                name: 'Robot Base',
                marker: { size: 15, color: 'red', symbol: 'diamond' }
            });

            // Add collision objects
            collisionObjects.forEach((obj, index) => {
                if (obj.type === 'Cuboid') {
                    let x = obj.position[0];
                    let y = obj.position[1];
                    let width = obj.dimensions[0];
                    let height = obj.dimensions[1];
                    
                    // Create rectangle
                    traces.push({
                        x: [x - width/2, x + width/2, x + width/2, x - width/2, x - width/2],
                        y: [y - height/2, y - height/2, y + height/2, y + height/2, y - height/2],
                        mode: 'lines',
                        type: 'scatter',
                        name: obj.name,
                        line: { color: `hsl(${index * 60}, 70%, 50%)`, width: 2 },
                        fill: 'tonexty',
                        fillcolor: `hsla(${index * 60}, 70%, 50%, 0.3)`
                    });
                }
            });

            let layout = {
                title: 'Collision Objects - Top-Down View (X-Y Plane)',
                xaxis: { title: 'X (m)', range: [workspaceBounds.x[0], workspaceBounds.x[1]] },
                yaxis: { title: 'Y (m)', range: [workspaceBounds.y[0], workspaceBounds.y[1]] },
                showlegend: true,
                width: 800,
                height: 600
            };

            Plotly.newPlot('top-down-plot', traces, layout);
        }

        // Create side view plot
        function createSideViewPlot() {
            let traces = [];
            
            // Add robot base
            traces.push({
                x: [0],
                y: [0],
                mode: 'markers',
                type: 'scatter',
                name: 'Robot Base',
                marker: { size: 15, color: 'red', symbol: 'diamond' }
            });

            // Add collision objects
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
                        line: { color: `hsl(${index * 60}, 70%, 50%)`, width: 2 },
                        fill: 'tonexty',
                        fillcolor: `hsla(${index * 60}, 70%, 50%, 0.3)`
                    });
                }
            });

            let layout = {
                title: 'Collision Objects - Side View (X-Z Plane)',
                xaxis: { title: 'X (m)', range: [workspaceBounds.x[0], workspaceBounds.x[1]] },
                yaxis: { title: 'Z (m)', range: [workspaceBounds.z[0], workspaceBounds.z[1]] },
                showlegend: true,
                width: 800,
                height: 600
            };

            Plotly.newPlot('side-view-plot', traces, layout);
        }

        // Create point cloud plot
        function createPointCloudPlot() {
            if (pointCloudData.length === 0) {
                document.getElementById('point-cloud-plot').innerHTML = '<p>No point cloud data available</p>';
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
                name: 'Point Cloud',
                marker: {
                    size: 2,
                    color: z,
                    colorscale: 'Viridis',
                    opacity: 0.6
                }
            };

            let layout = {
                title: 'Point Cloud Visualization (X-Y Projection)',
                xaxis: { title: 'X (m)' },
                yaxis: { title: 'Y (m)' },
                width: 800,
                height: 600
            };

            Plotly.newPlot('point-cloud-plot', [trace], layout);
        }

        // Create objects table
        function createObjectsTable() {
            let table = '<table border="1" style="width:100%; border-collapse: collapse;">';
            table += '<tr><th>Object</th><th>Position (X,Y,Z)</th><th>Dimensions (W,H,D)</th><th>Volume (m³)</th></tr>';
            
            collisionObjects.forEach(obj => {
                if (obj.type === 'Cuboid') {
                    let volume = obj.dimensions[0] * obj.dimensions[1] * obj.dimensions[2];
                    table += `<tr>
                        <td>${obj.name}</td>
                        <td>(${obj.position[0].toFixed(3)}, ${obj.position[1].toFixed(3)}, ${obj.position[2].toFixed(3)})</td>
                        <td>(${obj.dimensions[0].toFixed(3)}, ${obj.dimensions[1].toFixed(3)}, ${obj.dimensions[2].toFixed(3)})</td>
                        <td>${volume.toFixed(3)}</td>
                    </tr>`;
                }
            });
            table += '</table>';
            
            document.getElementById('objects-table').innerHTML = table;
        }

        // Capture new data
        function captureData() {
            document.getElementById('status').className = 'status info';
            document.getElementById('status').textContent = 'Capturing data from RealSense camera...';
            
            fetch('/capture_data')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        collisionObjects = data.collision_objects;
                        pointCloudData = data.point_cloud;
                        updateStats();
                        createTopDownPlot();
                        createSideViewPlot();
                        createPointCloudPlot();
                        createObjectsTable();
                        
                        document.getElementById('status').className = 'status success';
                        document.getElementById('status').textContent = `✅ Successfully captured data: ${collisionObjects.length} objects, ${pointCloudData.length} points`;
                    } else {
                        document.getElementById('status').className = 'status error';
                        document.getElementById('status').textContent = '❌ Failed to capture data: ' + data.error;
                    }
                })
                .catch(error => {
                    document.getElementById('status').className = 'status error';
                    document.getElementById('status').textContent = '❌ Error: ' + error;
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
                point_cloud: pointCloudData,
                workspace_bounds: workspaceBounds,
                timestamp: new Date().toISOString()
            };
            
            let blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            let url = URL.createObjectURL(blob);
            let a = document.createElement('a');
            a.href = url;
            a.download = 'collision_objects_data.json';
            a.click();
            URL.revokeObjectURL(url);
            
            document.getElementById('status').className = 'status success';
            document.getElementById('status').textContent = '✅ Data exported successfully';
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
    global collision_objects, point_cloud_data
    
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
        
        # Convert point cloud to list for JSON serialization
        point_cloud_data = pcd.tolist() if hasattr(pcd, 'tolist') else pcd
        
        # Cleanup
        camera.stop()
        motion_planner.disconnect_robot()
        
        return jsonify({
            'success': True,
            'collision_objects': collision_objects,
            'point_cloud': point_cloud_data
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
    print("🌐 Starting web visualization server...")
    print("📱 Open your web browser and go to: http://localhost:5000")
    print("📊 The web interface will show:")
    print("   • Real-time collision objects visualization")
    print("   • Point cloud data from RealSense")
    print("   • Interactive plots with Plotly")
    print("   • Export functionality for data")
    print("\n🔄 Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    run_server() 