#!/usr/bin/env python3
"""
Demo web server for collision objects visualization
Uses sample data to demonstrate the optimized interface
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

app = Flask(__name__)

# Sample data from our previous successful runs
SAMPLE_COLLISION_OBJECTS = [
    {
        'name': 'obs_0',
        'type': 'Cuboid',
        'position': [0.597, -0.291, 0.730],
        'dimensions': [0.769, 0.836, 0.795],
        'orientation': [-0.456, 0.550, -0.524, 0.464]
    },
    {
        'name': 'obs_1',
        'type': 'Cuboid',
        'position': [0.866, 0.059, 0.385],
        'dimensions': [0.188, 0.054, 0.115],
        'orientation': [-0.456, 0.550, -0.524, 0.464]
    },
    {
        'name': 'obs_2',
        'type': 'Cuboid',
        'position': [0.869, -0.100, 0.366],
        'dimensions': [0.117, 0.026, 0.035],
        'orientation': [-0.456, 0.550, -0.524, 0.464]
    },
    {
        'name': 'obs_3',
        'type': 'Cuboid',
        'position': [0.731, -0.236, 0.407],
        'dimensions': [0.049, 0.017, 0.040],
        'orientation': [-0.456, 0.550, -0.524, 0.464]
    },
    {
        'name': 'obs_4',
        'type': 'Cuboid',
        'position': [0.927, -0.806, 1.144],
        'dimensions': [0.326, 0.238, 0.071],
        'orientation': [-0.456, 0.550, -0.524, 0.464]
    }
]

# Generate sample point cloud data
def generate_sample_point_cloud():
    """Generate realistic sample point cloud data"""
    np.random.seed(42)  # For reproducible results
    
    # Generate points around the collision objects
    points = []
    
    # Add points around obs_0 (main object)
    n_points = 2000
    center = [0.597, -0.291, 0.730]
    size = [0.769, 0.836, 0.795]
    
    for _ in range(n_points):
        x = center[0] + np.random.normal(0, size[0]/4)
        y = center[1] + np.random.normal(0, size[1]/4)
        z = center[2] + np.random.normal(0, size[2]/4)
        points.append([x, y, z])
    
    # Add points around obs_1
    center = [0.866, 0.059, 0.385]
    size = [0.188, 0.054, 0.115]
    
    for _ in range(500):
        x = center[0] + np.random.normal(0, size[0]/4)
        y = center[1] + np.random.normal(0, size[1]/4)
        z = center[2] + np.random.normal(0, size[2]/4)
        points.append([x, y, z])
    
    # Add points around obs_4
    center = [0.927, -0.806, 1.144]
    size = [0.326, 0.238, 0.071]
    
    for _ in range(300):
        x = center[0] + np.random.normal(0, size[0]/4)
        y = center[1] + np.random.normal(0, size[1]/4)
        z = center[2] + np.random.normal(0, size[2]/4)
        points.append([x, y, z])
    
    # Add some random background points
    for _ in range(500):
        x = np.random.uniform(-0.5, 1.0)
        y = np.random.uniform(-0.8, 0.8)
        z = np.random.uniform(0.0, 1.2)
        points.append([x, y, z])
    
    return np.array(points)

# Global variables
collision_objects = []
point_cloud_data = []
workspace_bounds = {
    'x': [-0.5, 1.0],
    'y': [-0.8, 0.8],
    'z': [0.0, 1.2]
}

# Optimized HTML template (same as optimized version)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Collision Objects Visualization (Demo)</title>
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
        .demo-info { background: #e3f2fd; padding: 15px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #2196f3; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 RealSense Collision Objects Visualization (Demo)</h1>
            <p>High-performance visualization with sample data</p>
        </div>

        <div class="demo-info">
            <strong>📋 Demo Mode:</strong> This interface uses sample data to demonstrate the optimized visualization. 
            The RealSense camera integration is available in the full version.
        </div>

        <div class="controls">
            <button onclick="loadSampleData()" id="capture-btn">📊 Load Sample Data</button>
            <button onclick="refreshVisualization()" id="refresh-btn">🔄 Refresh</button>
            <button onclick="exportData()" id="export-btn">💾 Export Data</button>
            <button onclick="togglePerformance()" id="perf-btn">⚡ Performance Info</button>
        </div>

        <div id="status" class="status info">
            Ready to load sample data...
        </div>

        <div class="performance-info" id="perf-info" style="display: none;">
            <strong>Performance Optimizations:</strong><br>
            • Point cloud downsampling (max 5,000 points for display)<br>
            • Data caching to reduce server load<br>
            • Efficient JSON serialization<br>
            • Responsive grid layout<br>
            • Optimized plot rendering<br>
            • Sample data for demonstration
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

        // Load sample data
        function loadSampleData() {
            if (isCapturing) return;
            
            isCapturing = true;
            startTime = performance.now();
            
            document.getElementById('capture-btn').disabled = true;
            document.getElementById('capture-btn').textContent = '⏳ Loading...';
            document.getElementById('status').className = 'status loading';
            document.getElementById('status').innerHTML = '<span class="loading-spinner"></span> Loading sample data...';
            
            // Simulate loading time
            setTimeout(() => {
                fetch('/load_sample_data')
                    .then(response => response.json())
                    .then(data => {
                        endTime = performance.now();
                        const loadTime = ((endTime - startTime) / 1000).toFixed(2);
                        
                        if (data.success) {
                            collisionObjects = data.collision_objects;
                            pointCloudData = data.point_cloud;
                            
                            updateStats();
                            createTopDownPlot();
                            createSideViewPlot();
                            createPointCloudPlot();
                            createObjectsTable();
                            
                            document.getElementById('status').className = 'status success';
                            document.getElementById('status').innerHTML = `✅ Successfully loaded sample data in ${loadTime}s: ${collisionObjects.length} objects, ${pointCloudData.length} points`;
                        } else {
                            document.getElementById('status').className = 'status error';
                            document.getElementById('status').textContent = '❌ Failed to load sample data: ' + data.error;
                        }
                    })
                    .catch(error => {
                        document.getElementById('status').className = 'status error';
                        document.getElementById('status').textContent = '❌ Error: ' + error;
                    })
                    .finally(() => {
                        isCapturing = false;
                        document.getElementById('capture-btn').disabled = false;
                        document.getElementById('capture-btn').textContent = '📊 Load Sample Data';
                    });
            }, 1000); // Simulate 1 second loading time
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
                    load_time: ((endTime - startTime) / 1000).toFixed(2) + 's',
                    point_cloud_size: pointCloudData.length + ' points',
                    demo_mode: true
                }
            };
            
            let blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            let url = URL.createObjectURL(blob);
            let a = document.createElement('a');
            a.href = url;
            a.download = 'collision_objects_demo_data.json';
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

@app.route('/load_sample_data')
def load_sample_data():
    global collision_objects, point_cloud_data
    
    try:
        # Load sample collision objects
        collision_objects = SAMPLE_COLLISION_OBJECTS
        
        # Generate sample point cloud
        sample_pcd = generate_sample_point_cloud()
        
        # Sample for display (performance optimization)
        if len(sample_pcd) > 5000:
            indices = random.sample(range(len(sample_pcd)), 5000)
            sample_pcd = sample_pcd[indices]
        
        point_cloud_data = sample_pcd.tolist()
        
        return jsonify({
            'success': True,
            'collision_objects': collision_objects,
            'point_cloud': point_cloud_data,
            'original_point_count': len(sample_pcd),
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
    print("🎭 Starting DEMO web visualization server...")
    print("📱 Open your web browser and go to: http://localhost:5000")
    print("📊 Demo features:")
    print("   • Sample collision objects data")
    print("   • Generated point cloud visualization")
    print("   • Optimized performance")
    print("   • Responsive design")
    print("   • No RealSense camera required")
    print("\n🔄 Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    run_server() 