#!/usr/bin/env python3
# ============================================
# COMPLETE ORTHOMOSAIC DETECTION PIPELINE (FIXED)
# ============================================

import os
import sys
import subprocess
import numpy as np
import cv2
from pathlib import Path
import json
from datetime import datetime
import time
import gc
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================
# GPU SETUP
# ============================================
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🎮 Using device: {device.upper()}")
if device == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from ultralytics import YOLO

# ============================================
# GEOSPATIAL SETUP (FIXED - NO PIP GDAL)
# ============================================
def setup_gdal():
    """Use system GDAL instead of pip GDAL"""
    # Add system GDAL path if needed
    sys.path.insert(0, '/usr/lib/python3/dist-packages')
    
    try:
        from osgeo import gdal, osr, ogr
        from osgeo import gdal_array
        print("✅ GDAL available - Full geospatial support enabled")
        return True
    except ImportError:
        print("❌ GDAL not properly installed.")
        print("Run: sudo apt install -y python3-gdal")
        print("And: pip uninstall gdal -y")
        return False

if not setup_gdal():
    sys.exit(1)

from osgeo import gdal, osr, ogr

# Shapely
try:
    from shapely.geometry import Polygon, Point, box
    from shapely.ops import unary_union
    print("✅ Shapely available")
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "shapely"], check=False)
    from shapely.geometry import Polygon, Point, box
    from shapely.ops import unary_union
    print("✅ Shapely installed")

# ============================================
# CONFIGURATION
# ============================================
ORTHOMOSAIC_PATH = "Chorkor1clipped.tif"
MODEL_PATH = "best.pt"
OUTPUT_DIR = "DetectionResults"
TILE_SIZE = 640
OVERLAP = 0.15
CONFIDENCE_THRESHOLD = 0.15
IOU_THRESHOLD = 0.45
CHECKPOINT_INTERVAL = 100  # Save every 100 tiles
MERGE_IOU_THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"""
╔══════════════════════════════════════════════════════════════╗
║     COMPLETE PIPELINE + CHECKPOINTING + ANALYTICS             ║
╠══════════════════════════════════════════════════════════════╣
║ Orthomosaic: {os.path.basename(ORTHOMOSAIC_PATH)}
║ Size: {os.path.getsize(ORTHOMOSAIC_PATH) / (1024**3):.2f} GB
║ Output: {OUTPUT_DIR}
║ Checkpoint: Every {CHECKPOINT_INTERVAL} tiles
║ Device: {device.upper()}
╚══════════════════════════════════════════════════════════════╝
""")


# ============================================
# CHECKPOINT MANAGER
# ============================================
class CheckpointManager:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save(self, detections, tile_count, metadata=None):
        """Save checkpoint with detections"""
        checkpoint = {
            'detections': detections,
            'tiles_processed': tile_count,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        filename = os.path.join(self.checkpoint_dir, f"checkpoint_{tile_count:06d}.json")
        with open(filename, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        return filename
    
    def save_intermediate_results(self, detections, tile_count, tiler_metadata):
        """Save intermediate CSV/JSON for current progress"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save intermediate JSON
        result = {
            'metadata': {
                'tiles_processed': tile_count,
                'detections_count': len(detections),
                'timestamp': timestamp,
                'is_complete': False,
                **tiler_metadata
            },
            'detections': detections
        }
        
        json_path = os.path.join(self.output_dir, f"intermediate_{tile_count:06d}_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Save intermediate CSV
        csv_path = os.path.join(self.output_dir, f"intermediate_{tile_count:06d}_{timestamp}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'confidence', 'center_x', 'center_y', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'area'])
            for i, det in enumerate(detections):
                writer.writerow([
                    i, det['confidence'],
                    det['center_geo'][0], det['center_geo'][1],
                    det['bbox_geo'][0], det['bbox_geo'][1],
                    det['bbox_geo'][2], det['bbox_geo'][3],
                    det['area_pixel']
                ])
        
        return json_path, csv_path
    
    def load_latest(self):
        """Load latest checkpoint"""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoints = sorted([f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.json')])
        if checkpoints:
            latest = checkpoints[-1]
            with open(os.path.join(self.checkpoint_dir, latest), 'r') as f:
                return json.load(f)
        return None


# ============================================
# GEOSPATIAL TILER
# ============================================
class GeospatialTiler:
    def __init__(self, image_path, tile_size=640, overlap=0.15):
        self.image_path = image_path
        self.tile_size = tile_size
        self.stride = int(tile_size * (1 - overlap))
        
        self.dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
        self.width = self.dataset.RasterXSize
        self.height = self.dataset.RasterYSize
        self.bands = self.dataset.RasterCount
        self.geotransform = self.dataset.GetGeoTransform()
        
        projection = self.dataset.GetProjection()
        self.srs = osr.SpatialReference()
        self.srs.ImportFromWkt(projection)
        
        self.x_min = self.geotransform[0]
        self.y_max = self.geotransform[3]
        self.x_max = self.x_min + self.geotransform[1] * self.width
        self.y_min = self.y_max + self.geotransform[5] * self.height
        
        self.pixel_width = abs(self.geotransform[1])
        self.pixel_height = abs(self.geotransform[5])
        
        self.crs_epsg = self.srs.GetAuthorityCode(None) if self.srs.GetAuthorityName(None) == 'EPSG' else None
        self.crs_proj4 = self.srs.ExportToProj4()
        
        print(f"\n📂 Orthomosaic loaded:")
        print(f"   Dimensions: {self.width:,} x {self.height:,} pixels")
        print(f"   Bands: {self.bands}")
        print(f"   EPSG: {self.crs_epsg}")
        print(f"   Pixel size: {self.pixel_width:.3f}m x {self.pixel_height:.3f}m")
    
    def get_metadata(self):
        """Return tiler metadata for output"""
        return {
            'dimensions': [self.width, self.height],
            'bands': self.bands,
            'epsg': self.crs_epsg,
            'proj4': self.crs_proj4,
            'bounds': {
                'left': self.x_min,
                'bottom': self.y_min,
                'right': self.x_max,
                'top': self.y_max
            },
            'pixel_size': {
                'x': self.pixel_width,
                'y': self.pixel_height
            }
        }
    
    def pixel_to_geo(self, x, y):
        geo_x = self.geotransform[0] + x * self.geotransform[1] + y * self.geotransform[2]
        geo_y = self.geotransform[3] + x * self.geotransform[4] + y * self.geotransform[5]
        return geo_x, geo_y
    
    def get_total_tiles(self):
        tiles_x = (self.width - self.tile_size) // self.stride + 1
        tiles_y = (self.height - self.tile_size) // self.stride + 1
        return tiles_x * tiles_y
    
    def stream_tiles(self, start_tile=0):
        total_tiles = self.get_total_tiles()
        tile_count = 0
        
        pbar = tqdm(total=total_tiles - start_tile, desc="Processing tiles", unit="tile")
        
        for y in range(0, self.height - self.tile_size + 1, self.stride):
            for x in range(0, self.width - self.tile_size + 1, self.stride):
                if tile_count < start_tile:
                    tile_count += 1
                    continue
                
                # Read tile
                tile_data = self.dataset.ReadAsArray(int(x), int(y), self.tile_size, self.tile_size)
                
                if tile_data is None:
                    tile_count += 1
                    pbar.update(1)
                    continue
                
                # Format image
                if len(tile_data.shape) == 3:
                    tile_img = np.transpose(tile_data, (1, 2, 0))
                else:
                    tile_img = np.stack([tile_data] * 3, axis=-1)
                if tile_img.shape[2] >= 3:
                    tile_img = tile_img[:, :, :3]
                
                # Calculate bounds
                geo_x_min, geo_y_max = self.pixel_to_geo(x, y)
                geo_x_max, geo_y_min = self.pixel_to_geo(x + self.tile_size, y + self.tile_size)
                
                yield {
                    'image': tile_img,
                    'x_offset': x,
                    'y_offset': y,
                    'tile_index': tile_count,
                    'bounds': {
                        'x_min': geo_x_min, 'y_min': geo_y_min,
                        'x_max': geo_x_max, 'y_max': geo_y_max
                    }
                }
                
                tile_count += 1
                pbar.update(1)
        
        pbar.close()
    
    def close(self):
        if self.dataset:
            self.dataset = None


# ============================================
# DETECTOR
# ============================================
class PlasticDetector:
    def __init__(self, model_path, conf_threshold=0.15):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
    
    def detect_on_tile(self, tile_info):
        tile_img = tile_info['image']
        x_offset = tile_info['x_offset']
        y_offset = tile_info['y_offset']
        bounds = tile_info['bounds']
        
        results = self.model.predict(tile_img, conf=self.conf_threshold, iou=0.45, 
                                     verbose=False, device=device)
        
        detections = []
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box
                
                global_x1 = x_offset + x1
                global_y1 = y_offset + y1
                global_x2 = x_offset + x2
                global_y2 = y_offset + y2
                
                geo_x1 = bounds['x_min'] + (x1 / TILE_SIZE) * (bounds['x_max'] - bounds['x_min'])
                geo_y1 = bounds['y_max'] - (y1 / TILE_SIZE) * (bounds['y_max'] - bounds['y_min'])
                geo_x2 = bounds['x_min'] + (x2 / TILE_SIZE) * (bounds['x_max'] - bounds['x_min'])
                geo_y2 = bounds['y_max'] - (y2 / TILE_SIZE) * (bounds['y_max'] - bounds['y_min'])
                
                detections.append({
                    'bbox_pixel': [float(global_x1), float(global_y1), float(global_x2), float(global_y2)],
                    'bbox_geo': [geo_x1, geo_y1, geo_x2, geo_y2],
                    'center_pixel': [(global_x1 + global_x2)/2, (global_y1 + global_y2)/2],
                    'center_geo': [(geo_x1 + geo_x2)/2, (geo_y1 + geo_y2)/2],
                    'confidence': float(conf),
                    'class': 'plastic-waste',
                    'area_pixel': float((global_x2 - global_x1) * (global_y2 - global_y1)),
                    'area_geo': float(abs((geo_x2 - geo_x1) * (geo_y2 - geo_y1)))
                })
        
        return detections
    
    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0
    
    def merge_detections(self, detections, iou_threshold=0.5):
        if len(detections) < 2:
            return detections
        
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            box1 = det1['bbox_pixel']
            overlapping = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                if self.calculate_iou(box1, det2['bbox_pixel']) > iou_threshold:
                    overlapping.append(det2)
                    used.add(j)
            
            if len(overlapping) > 1:
                all_x1 = [d['bbox_pixel'][0] for d in overlapping]
                all_y1 = [d['bbox_pixel'][1] for d in overlapping]
                all_x2 = [d['bbox_pixel'][2] for d in overlapping]
                all_y2 = [d['bbox_pixel'][3] for d in overlapping]
                
                all_geo_x1 = [d['bbox_geo'][0] for d in overlapping]
                all_geo_y1 = [d['bbox_geo'][1] for d in overlapping]
                all_geo_x2 = [d['bbox_geo'][2] for d in overlapping]
                all_geo_y2 = [d['bbox_geo'][3] for d in overlapping]
                
                merged.append({
                    'bbox_pixel': [min(all_x1), min(all_y1), max(all_x2), max(all_y2)],
                    'bbox_geo': [min(all_geo_x1), min(all_geo_y1), max(all_geo_x2), max(all_geo_y2)],
                    'center_pixel': [(min(all_x1) + max(all_x2))/2, (min(all_y1) + max(all_y2))/2],
                    'center_geo': [(min(all_geo_x1) + max(all_geo_x2))/2, (min(all_geo_y1) + max(all_geo_y2))/2],
                    'confidence': float(np.mean([d['confidence'] for d in overlapping])),
                    'class': 'plastic-waste',
                    'area_pixel': float((max(all_x2) - min(all_x1)) * (max(all_y2) - min(all_y1))),
                    'area_geo': float(abs((max(all_geo_x2) - min(all_geo_x1)) * (max(all_geo_y2) - min(all_geo_y1)))),
                    'merged_from': len(overlapping)
                })
            else:
                merged.append(det1)
        
        return merged


# ============================================
# ANALYTICS AND VISUALIZATION
# ============================================
class AnalyticsGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
    
    def generate_all(self, detections, metadata, timestamp):
        """Generate all analytics and visualizations"""
        print("\n" + "="*60)
        print("📊 GENERATING ANALYTICS AND VISUALIZATIONS")
        print("="*60)
        
        if len(detections) == 0:
            print("⚠️ No detections to analyze")
            return
        
        # Calculate statistics
        stats = self.calculate_statistics(detections, metadata)
        
        # Create visualizations
        vis_path = self.create_visualizations(detections, stats, metadata, timestamp)
        
        # Save statistics JSON
        stats_path = os.path.join(self.output_dir, f"statistics_{timestamp}.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Statistics saved: {stats_path}")
        
        return stats, vis_path
    
    def calculate_statistics(self, detections, metadata):
        """Calculate comprehensive statistics"""
        confidences = [d['confidence'] for d in detections]
        areas_pixel = [d['area_pixel'] for d in detections]
        areas_geo = [d['area_geo'] for d in detections]
        
        total_pixels = metadata['dimensions'][0] * metadata['dimensions'][1]
        
        stats = {
            'detection_counts': {
                'total': len(detections),
                'merged_detections': sum(1 for d in detections if d.get('merged_from', 1) > 1)
            },
            'confidence': {
                'mean': float(np.mean(confidences)),
                'median': float(np.median(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'percentiles': {
                    '25': float(np.percentile(confidences, 25)),
                    '50': float(np.percentile(confidences, 50)),
                    '75': float(np.percentile(confidences, 75)),
                    '90': float(np.percentile(confidences, 90)),
                    '95': float(np.percentile(confidences, 95))
                }
            },
            'area_pixels': {
                'total': float(np.sum(areas_pixel)),
                'mean': float(np.mean(areas_pixel)),
                'median': float(np.median(areas_pixel)),
                'std': float(np.std(areas_pixel)),
                'min': float(np.min(areas_pixel)),
                'max': float(np.max(areas_pixel))
            },
            'area_geo': {
                'total_sqm': float(np.sum(areas_geo)),
                'mean_sqm': float(np.mean(areas_geo)),
                'median_sqm': float(np.median(areas_geo)),
                'min_sqm': float(np.min(areas_geo)),
                'max_sqm': float(np.max(areas_geo))
            },
            'spatial': {
                'coverage_percentage': float(np.sum(areas_pixel) / total_pixels * 100),
                'detection_density_per_megapixel': float(len(detections) / total_pixels * 1e6),
                'bbox_centers': {
                    'x_mean': float(np.mean([d['center_geo'][0] for d in detections])),
                    'y_mean': float(np.mean([d['center_geo'][1] for d in detections]))
                }
            }
        }
        
        # Confidence distribution bins
        bins = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
        hist, _ = np.histogram(confidences, bins=bins)
        stats['confidence']['distribution'] = {f"{bins[i]:.2f}-{bins[i+1]:.2f}": int(hist[i]) for i in range(len(hist))}
        
        return stats
    
    def create_visualizations(self, detections, stats, metadata, timestamp):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Plastic Waste Detection - Comprehensive Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        confidences = [d['confidence'] for d in detections]
        areas = [d['area_pixel'] for d in detections]
        centers_x = [d['center_geo'][0] for d in detections]
        centers_y = [d['center_geo'][1] for d in detections]
        
        # Plot 1: Confidence Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(confidences, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(x=CONFIDENCE_THRESHOLD, color='red', linestyle='--', label=f'Threshold ({CONFIDENCE_THRESHOLD})')
        ax1.axvline(x=stats['confidence']['median'], color='green', linestyle='--', label=f"Median ({stats['confidence']['median']:.3f})")
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Count')
        ax1.set_title('Detection Confidence Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Area Distribution (log scale)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(np.log10(areas), bins=20, color='coral', edgecolor='black', alpha=0.7)
        ax2.axvline(x=np.log10(stats['area_pixels']['median']), color='green', linestyle='--', label='Median')
        ax2.set_xlabel('log10(Area in pixels)')
        ax2.set_ylabel('Count')
        ax2.set_title('Detection Size Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Confidence vs Area Scatter
        ax3 = fig.add_subplot(gs[0, 2])
        scatter = ax3.scatter(np.log10(areas), confidences, c=confidences, cmap='YlOrRd', 
                             s=20, alpha=0.6, edgecolors='black', linewidth=0.3)
        ax3.set_xlabel('log10(Area in pixels)')
        ax3.set_ylabel('Confidence')
        ax3.set_title('Confidence vs Detection Size')
        plt.colorbar(scatter, ax=ax3, label='Confidence')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Spatial Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        scatter = ax4.scatter(centers_x, centers_y, c=confidences, cmap='YlOrRd', 
                             s=20, alpha=0.6, edgecolors='black', linewidth=0.3)
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        ax4.set_title('Detection Locations')
        plt.colorbar(scatter, ax=ax4, label='Confidence')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Spatial Heatmap
        ax5 = fig.add_subplot(gs[1, 1])
        hist = ax5.hist2d(centers_x, centers_y, bins=50, cmap='hot')
        ax5.set_xlabel('Longitude')
        ax5.set_ylabel('Latitude')
        ax5.set_title('Detection Density Heatmap')
        plt.colorbar(hist[3], ax=ax5, label='Count')
        
        # Plot 6: Cumulative Confidence
        ax6 = fig.add_subplot(gs[1, 2])
        sorted_conf = np.sort(confidences)
        cumulative = np.arange(1, len(sorted_conf) + 1) / len(sorted_conf) * 100
        ax6.plot(sorted_conf, cumulative, linewidth=2, color='purple')
        ax6.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
        ax6.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90%')
        ax6.set_xlabel('Confidence Threshold')
        ax6.set_ylabel('Cumulative % of Detections')
        ax6.set_title('Cumulative Confidence Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Plot 7-9: Summary Statistics
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        summary_text = f"""
        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║                         DETECTION SUMMARY REPORT                               ║
        ╠══════════════════════════════════════════════════════════════════════════════╣
        ║                                                                                ║
        ║  📊 DETECTION COUNTS                                                            ║
        ║     • Total Detections:        {stats['detection_counts']['total']:,}                                         ║
        ║     • Merged Detections:       {stats['detection_counts']['merged_detections']:,}                                         ║
        ║                                                                                ║
        ║  🎯 CONFIDENCE STATISTICS                                                       ║
        ║     • Mean:                    {stats['confidence']['mean']:.4f}                                              ║
        ║     • Median:                  {stats['confidence']['median']:.4f}                                              ║
        ║     • Std Dev:                 {stats['confidence']['std']:.4f}                                              ║
        ║     • Range:                   [{stats['confidence']['min']:.4f}, {stats['confidence']['max']:.4f}]                                    ║
        ║                                                                                ║
        ║  📐 AREA STATISTICS (PIXELS)                                                    ║
        ║     • Total Area:              {stats['area_pixels']['total']:,.0f}                                           ║
        ║     • Mean Area:               {stats['area_pixels']['mean']:,.0f}                                           ║
        ║     • Median Area:             {stats['area_pixels']['median']:,.0f}                                           ║
        ║                                                                                ║
        ║  🌍 GEOGRAPHIC STATISTICS                                                       ║
        ║     • Total Area (m²):         {stats['area_geo']['total_sqm']:,.2f}                                         ║
        ║     • Mean Area (m²):          {stats['area_geo']['mean_sqm']:,.2f}                                         ║
        ║     • Coverage:                {stats['spatial']['coverage_percentage']:.4f}%                                          ║
        ║     • Density:                 {stats['spatial']['detection_density_per_megapixel']:.2f} / MP                                    ║
        ║                                                                                ║
        ║  📁 METADATA                                                                     ║
        ║     • Image Dimensions:        {metadata['dimensions'][0]:,} x {metadata['dimensions'][1]:,}                                 ║
        ║     • EPSG:                    {metadata['epsg']}                                                     ║
        ║     • Pixel Size:              {metadata['pixel_size']['x']:.3f}m x {metadata['pixel_size']['y']:.3f}m                          ║
        ║     • Processed Date:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                   ║
        ║                                                                                ║
        ╚══════════════════════════════════════════════════════════════════════════════╝
        """
        
        ax7.text(0.5, 0.5, summary_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        vis_path = os.path.join(self.output_dir, f"comprehensive_analysis_{timestamp}.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Visualizations saved: {vis_path}")
        return vis_path


# ============================================
# FINAL RESULTS EXPORTER
# ============================================
class FinalExporter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
    
    def export_all(self, detections, metadata, stats, timestamp):
        """Export all final results"""
        print("\n" + "="*60)
        print("💾 EXPORTING FINAL RESULTS")
        print("="*60)
        
        # Complete JSON
        result = {
            'metadata': {
                **metadata,
                'processing_completed': datetime.now().isoformat(),
                'processing_time_hours': metadata.get('processing_time_hours', 0),
                'total_detections': len(detections),
                'confidence_threshold': CONFIDENCE_THRESHOLD,
                'tile_size': TILE_SIZE,
                'overlap': OVERLAP
            },
            'statistics': stats,
            'detections': detections
        }
        
        json_path = os.path.join(self.output_dir, f"FINAL_COMPLETE_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✅ Complete JSON: {json_path}")
        
        # GeoJSON
        features = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox_geo']
            feature = {
                'type': 'Feature',
                'id': i,
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]]
                },
                'properties': {
                    'id': i,
                    'confidence': det['confidence'],
                    'class': det['class'],
                    'area_pixel': det['area_pixel'],
                    'area_geo_sqm': det['area_geo'],
                    'center_geo_x': det['center_geo'][0],
                    'center_geo_y': det['center_geo'][1],
                    'merged_from': det.get('merged_from', 1)
                }
            }
            features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features,
            'metadata': {
                'source': ORTHOMOSAIC_PATH,
                'epsg': metadata['epsg'],
                'total_detections': len(detections),
                'timestamp': timestamp
            }
        }
        
        geojson_path = os.path.join(self.output_dir, f"FINAL_detections_{timestamp}.geojson")
        with open(geojson_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        print(f"✅ GeoJSON: {geojson_path}")
        
        # CSV
        csv_path = os.path.join(self.output_dir, f"FINAL_detections_{timestamp}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'confidence', 'center_x', 'center_y', 
                            'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                            'area_pixel', 'area_geo_sqm', 'merged_from'])
            for i, det in enumerate(detections):
                writer.writerow([
                    i, det['confidence'],
                    det['center_geo'][0], det['center_geo'][1],
                    det['bbox_geo'][0], det['bbox_geo'][1],
                    det['bbox_geo'][2], det['bbox_geo'][3],
                    det['area_pixel'], det['area_geo'],
                    det.get('merged_from', 1)
                ])
        print(f"✅ CSV: {csv_path}")
        
        return json_path, geojson_path, csv_path


# ============================================
# MAIN PIPELINE
# ============================================
# Add this import at the top
import cv2

# Add this configuration
SAVE_TILE_IMAGES = True  # Set to False to disable
TILE_IMAGES_DIR = os.path.join(OUTPUT_DIR, "tile_images")
if SAVE_TILE_IMAGES:
    os.makedirs(TILE_IMAGES_DIR, exist_ok=True)


# ============================================
# TILE IMAGE SAVER (NEW CLASS)
# ============================================
class TileImageSaver:
    """Save tile images with detection bounding boxes"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.tiles_with_detections_dir = os.path.join(output_dir, "tiles_with_detections")
        self.tiles_no_detections_dir = os.path.join(output_dir, "tiles_no_detections")
        
        if SAVE_TILE_IMAGES:
            os.makedirs(self.tiles_with_detections_dir, exist_ok=True)
            # Don't create no_detections dir unless needed
    
    def get_color(self, confidence):
        """Get color based on confidence"""
        if confidence >= 0.5:
            return (0, 255, 0)  # Green - High confidence
        elif confidence >= 0.25:
            return (0, 255, 255)  # Yellow - Medium confidence
        else:
            return (0, 0, 255)  # Red - Low confidence
    
    def save_tile_with_boxes(self, tile_img, detections, tile_index, x_offset, y_offset):
        """
        Save tile image with bounding boxes drawn
        
        Args:
            tile_img: Original tile image (numpy array)
            detections: List of detections for this tile
            tile_index: Index of the tile
            x_offset, y_offset: Tile offset in original image
        """
        if not SAVE_TILE_IMAGES:
            return
        
        # Make a copy to draw on
        img_with_boxes = tile_img.copy()
        
        # Draw each detection
        for det in detections:
            # Get tile-relative coordinates
            global_bbox = det['bbox_pixel']
            x1 = global_bbox[0] - x_offset
            y1 = global_bbox[1] - y_offset
            x2 = global_bbox[2] - x_offset
            y2 = global_bbox[3] - y_offset
            
            confidence = det['confidence']
            color = self.get_color(confidence)
            
            # Draw rectangle
            cv2.rectangle(img_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw confidence label
            label = f"{confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_with_boxes, (int(x1), int(y1) - label_h - 4), 
                         (int(x1) + label_w + 4, int(y1)), color, -1)
            cv2.putText(img_with_boxes, label, (int(x1) + 2, int(y1) - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add tile info text
        info_text = f"Tile: {tile_index:06d} | Offset: ({x_offset}, {y_offset}) | Detections: {len(detections)}"
        cv2.putText(img_with_boxes, info_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img_with_boxes, info_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Save image
        if len(detections) > 0:
            # Save in tiles_with_detections folder
            filename = f"tile_{tile_index:06d}_x{x_offset}_y{y_offset}_dets{len(detections)}.jpg"
            save_path = os.path.join(self.tiles_with_detections_dir, filename)
        else:
            # Only save 1% of empty tiles to save space
            if tile_index % 100 == 0:
                os.makedirs(self.tiles_no_detections_dir, exist_ok=True)
                filename = f"tile_{tile_index:06d}_x{x_offset}_y{y_offset}_empty.jpg"
                save_path = os.path.join(self.tiles_no_detections_dir, filename)
            else:
                return
        
        cv2.imwrite(save_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    
    def create_mosaic(self, max_tiles=100):
        """Create a mosaic of tiles with detections"""
        import glob
        
        detection_files = glob.glob(os.path.join(self.tiles_with_detections_dir, "*.jpg"))
        if not detection_files:
            return
        
        # Sort by tile index
        detection_files = sorted(detection_files)[:max_tiles]
        
        if not detection_files:
            return
        
        # Calculate grid size
        n_images = len(detection_files)
        grid_size = int(np.ceil(np.sqrt(n_images)))
        
        # Read first image to get size
        first_img = cv2.imread(detection_files[0])
        h, w = first_img.shape[:2]
        
        # Create mosaic
        mosaic = np.zeros((h * grid_size, w * grid_size, 3), dtype=np.uint8)
        
        for idx, filepath in enumerate(detection_files):
            if idx >= grid_size * grid_size:
                break
            img = cv2.imread(filepath)
            row = idx // grid_size
            col = idx % grid_size
            mosaic[row*h:(row+1)*h, col*w:(col+1)*w] = img
        
        # Save mosaic
        mosaic_path = os.path.join(self.output_dir, "tile_mosaic.jpg")
        cv2.imwrite(mosaic_path, mosaic)
        print(f"✅ Tile mosaic saved: {mosaic_path}")


# ============================================
# UPDATED MAIN PIPELINE WITH TILE SAVING
# ============================================
def main():
    checkpoint_mgr = CheckpointManager(OUTPUT_DIR)
    tile_saver = TileImageSaver(OUTPUT_DIR)  # NEW
    
    # Try to resume from checkpoint
    checkpoint = checkpoint_mgr.load_latest()
    all_detections = []
    start_tile = 0
    
    if checkpoint:
        all_detections = checkpoint.get('detections', [])
        start_tile = checkpoint.get('tiles_processed', 0)
        print(f"\n✅ Resumed from checkpoint!")
        print(f"   Tiles processed: {start_tile:,}")
        print(f"   Detections so far: {len(all_detections):,}")
        
        response = input("\nContinue from checkpoint? (y/n): ").strip().lower()
        if response != 'y':
            all_detections = []
            start_tile = 0
            print("   Starting fresh...")
    
    # Initialize
    tiler = GeospatialTiler(ORTHOMOSAIC_PATH, TILE_SIZE, OVERLAP)
    detector = PlasticDetector(MODEL_PATH, CONFIDENCE_THRESHOLD)
    
    total_tiles = tiler.get_total_tiles()
    print(f"\n📊 Total tiles: {total_tiles:,}")
    print(f"   Starting from: {start_tile:,}")
    if SAVE_TILE_IMAGES:
        print(f"   📸 Saving tile images to: {TILE_IMAGES_DIR}")
    
    start_time = time.time()
    tile_count = start_tile
    last_checkpoint_time = time.time()
    tiles_with_detections = 0
    
    try:
        for tile_info in tiler.stream_tiles(start_tile=start_tile):
            detections = detector.detect_on_tile(tile_info)
            all_detections.extend(detections)
            
            # NEW: Save tile image with bounding boxes
            if SAVE_TILE_IMAGES:
                tile_saver.save_tile_with_boxes(
                    tile_info['image'], 
                    detections, 
                    tile_info['tile_index'],
                    tile_info['x_offset'],
                    tile_info['y_offset']
                )
                if len(detections) > 0:
                    tiles_with_detections += 1
            
            tile_count += 1
            
            # Save checkpoint and intermediate results
            if tile_count % CHECKPOINT_INTERVAL == 0:
                checkpoint_mgr.save(all_detections, tile_count)
                
                # Also save intermediate CSV/JSON
                tiler_metadata = tiler.get_metadata()
                intermediate_json, intermediate_csv = checkpoint_mgr.save_intermediate_results(
                    all_detections, tile_count, tiler_metadata
                )
                
                checkpoint_time = time.time() - last_checkpoint_time
                print(f"\n   💾 Checkpoint saved at tile {tile_count:,}")
                print(f"      Detections: {len(all_detections):,}")
                print(f"      Tiles with detections: {tiles_with_detections:,}")
                print(f"      Time since last checkpoint: {checkpoint_time:.1f}s")
                last_checkpoint_time = time.time()
            
            # Free memory
            del tile_info
            gc.collect()
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted! Saving checkpoint...")
        checkpoint_mgr.save(all_detections, tile_count)
        tiler_metadata = tiler.get_metadata()
        checkpoint_mgr.save_intermediate_results(all_detections, tile_count, tiler_metadata)
        print(f"   Checkpoint saved at tile {tile_count:,}")
        tiler.close()
        
        # Create mosaic of tiles processed so far
        if SAVE_TILE_IMAGES:
            tile_saver.create_mosaic()
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        checkpoint_mgr.save(all_detections, tile_count)
        print(f"   Emergency checkpoint saved at tile {tile_count:,}")
        tiler.close()
        raise
    
    tiler.close()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Processing time: {elapsed/3600:.1f} hours ({elapsed/60:.0f} minutes)")
    print(f"✅ Found {len(all_detections):,} raw detections")
    print(f"📸 Tiles with detections: {tiles_with_detections:,}")
    
    # Create mosaic of tiles with detections
    if SAVE_TILE_IMAGES:
        print(f"\n🖼️ Creating tile mosaic...")
        tile_saver.create_mosaic()
    
    # Merge overlapping detections
    print(f"\n🔄 Merging overlapping detections...")
    merged = detector.merge_detections(all_detections, MERGE_IOU_THRESHOLD)
    print(f"   Merged to {len(merged):,} unique detections")
    
    # Get tiler metadata
    tiler_metadata = tiler.get_metadata()
    tiler_metadata['processing_time_hours'] = elapsed/3600
    tiler_metadata['tiles_processed'] = tile_count
    tiler_metadata['tiles_with_detections'] = tiles_with_detections  # NEW
    
    # Generate analytics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analytics = AnalyticsGenerator(OUTPUT_DIR)
    stats, vis_path = analytics.generate_all(merged, tiler_metadata, timestamp)
    
    # Export final results
    exporter = FinalExporter(OUTPUT_DIR)
    json_path, geojson_path, csv_path = exporter.export_all(merged, tiler_metadata, stats, timestamp)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)
    print(f"""
    📁 Results saved to: {OUTPUT_DIR}/
    
    📄 Final Files:
       • {os.path.basename(json_path)}
       • {os.path.basename(geojson_path)}
       • {os.path.basename(csv_path)}
       • {os.path.basename(vis_path)}
       • {os.path.basename(os.path.join(OUTPUT_DIR, f'statistics_{timestamp}.json'))}
    
    🖼️ Tile Images:
       • Tiles with detections: {OUTPUT_DIR}/tiles_with_detections/
       • Tile mosaic: {OUTPUT_DIR}/tile_mosaic.jpg
    
    📁 Checkpoints: {OUTPUT_DIR}/checkpoints/
       • {len([f for f in os.listdir(os.path.join(OUTPUT_DIR, 'checkpoints')) if f.endswith('.json')])} checkpoint files
    
    🎯 Final Statistics:
       • Tiles processed: {tile_count:,}
       • Tiles with detections: {tiles_with_detections:,} ({tiles_with_detections/tile_count*100:.1f}%)
       • Raw detections: {len(all_detections):,}
       • Final detections: {len(merged):,}
       • Total time: {elapsed/3600:.1f} hours
       • Mean confidence: {stats['confidence']['mean']:.3f}
       • Total affected area: {stats['area_geo']['total_sqm']:.2f} m²
    """)


if __name__ == "__main__":
    main()