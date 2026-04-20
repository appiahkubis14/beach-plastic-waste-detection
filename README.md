# PlastiScan v2 — Multi-Survey Drone Waste Detection

A complete end-to-end system for detecting plastic waste in drone aerial imagery using deep learning (YOLO), with an interactive web-based visualization dashboard.

## Overview

PlastiScan processes high-resolution drone orthomosaic images to automatically detect and localize plastic waste deposits. The system uses:

- **YOLO (Ultralytics)** - Object detection model for identifying waste in aerial imagery
- **GDAL** - Geospatial library for handling GeoTIFF orthomosaics
- **Leaflet** - Interactive web map for visualization
- **IndexedDB** - Browser-based data persistence

## Project Structure

```
workflow/
├── index.html                 # Web-based visualization dashboard
├── resume_pipeline.py         # Single orthomosaic processing pipeline
├── complete_pipeline.py       # Multi-orthomosaic processing pipeline
├── best.pt                    # YOLO model weights
├── FINAL_detections_20260418_202512.geojson  # Sample detection results
└── README.md                  # This file
```

## System Components

### 1. Detection Pipeline (`resume_pipeline.py`, `complete_pipeline.py`)

The Python backend processes drone orthomosaic images:

| Feature | Description |
|---------|-------------|
| **Tiled Processing** | Splits large orthomosaics into 640x640 tiles with 15% overlap |
| **Checkpointing** | Saves progress every 100 tiles for resume capability |
| **Detection** | YOLO object detection with configurable confidence threshold (default: 0.15) |
| **NMS/Merging** | Merges overlapping detections using IOU threshold (default: 0.5) |
| **Analytics** | Generates statistics, charts, and visualizations |
| **Export** | Outputs GeoJSON, CSV, and PNG reports |

#### Configuration Parameters

```python
TILE_SIZE = 640          # Tile dimensions in pixels
OVERLAP = 0.15           # Tile overlap percentage
CONFIDENCE_THRESHOLD = 0.15  # Minimum detection confidence
IOU_THRESHOLD = 0.45    # NMS IOU threshold
MERGE_IOU_THRESHOLD = 0.5  # Detection merging threshold
CHECKPOINT_INTERVAL = 100  # Tiles between checkpoints
```

#### Usage

**Single orthomosaic:**
```bash
python resume_pipeline.py
```

**Multiple orthomosaics:**
```bash
# Edit ORTHOMOSAIC_PATHS in complete_pipeline.py first
python complete_pipeline.py
```

#### Output Files

Each processing run creates:

```
DetectionResults/
├── FINAL_COMPLETE_<timestamp>.json    # Complete results with metadata
├── FINAL_detections_<timestamp>.geojson # GeoJSON feature collection
├── FINAL_detections_<timestamp>.csv    # Tabular detection data
├── comprehensive_analysis_<timestamp>.png # Visualization dashboard
├── statistics_<timestamp>.json        # Statistical summary
├── tiles_with_detections/              # Tile images with bounding boxes
└── checkpoints/                        # Resume checkpoints
```

### 2. Visualization Dashboard (`index.html`)

The frontend provides an interactive interface for analyzing detection results:

#### Features

| Feature | Description |
|---------|-------------|
| **Multi-Survey Support** | Load and compare multiple detection datasets |
| **Interactive Map** | Leaflet-based map with multiple basemaps (Dark, Street, Satellite, Topo) |
| **Confidence Filtering** | Filter detections by confidence score (High >0.5, Med 0.25-0.5, Low <0.25) |
| **Area Filtering** | Filter by detection size in square meters |
| **Statistical Dashboard** | Charts for confidence distribution, area histogram, size breakdown |
| **Data Table** | Sortable, searchable detection table |
| **Drawing Tools** | Rectangle and polygon selection for custom analysis |
| **Export** | Export filtered data to CSV or GeoJSON |
| **Screenshot** | Capture map view as PNG image |
| **Persistence** | IndexedDB storage for sessions |

#### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `F` | Fit all detections in view |
| `D` | Toggle dark/light theme |
| `H` | Toggle heatmap |
| `C` | Toggle clustering |
| `E` | Export CSV |
| `G` | Export GeoJSON |
| `T` | Show data table |
| `S` | Show surveys panel |
| `[` | Collapse sidebar |
| `]` | Expand sidebar |
| `?` | Show keyboard shortcuts |

#### Data Format

The dashboard accepts GeoJSON files with the following structure:

```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "id": 0,
    "geometry": {
      "type": "Polygon",
      "coordinates": [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]]
    },
    "properties": {
      "id": 0,
      "confidence": 0.657,
      "class": "plastic-waste",
      "area_pixel": 33120.17,
      "area_geo_sqm": 2.66,
      "center_geo_x": 804886.05,
      "center_geo_y": 611307.71
    }
  }]
}
```

#### Coordinate Systems

- Input: UTM EPSG:32630 (auto-converted to WGS84 for display)
- The system uses Proj4js for coordinate transformation

## Installation

### Prerequisites

```bash
# Python dependencies
pip install numpy opencv-python torch ultralytics tqdm matplotlib shapely

# System dependencies (required for GDAL)
sudo apt install -y python3-gdal gdal-bin

# Web dependencies (loaded via CDN in index.html)
# - Leaflet 1.9.4
# - Chart.js 4.4.0
# - Proj4js 2.9.0
# - PapaParse 5.4.1
# - html2canvas 1.4.1
```

### Model

Place your trained YOLO model as `best.pt` in the project directory. The model should be trained to detect plastic waste objects.

## Workflow

1. **Capture** - Drone captures aerial imagery of target area
2. **Process** - Generate orthomosaic from drone images (using Agisoft, Pix4D, etc.)
3. **Detect** - Run detection pipeline on orthomosaic
4. **Visualize** - Load results in PlastiScan dashboard for analysis
5. **Export** - Export filtered results for further analysis or reporting

## Detection Statistics

Sample detection results stored in `FINAL_detections_20260418_202512.geojson`:
- Contains thousands of plastic waste detections
- Each detection includes confidence score, area, and geographic coordinates

## Technical Details

### Memory Management
- Tiles are processed sequentially with garbage collection
- Checkpoint system prevents data loss on interruption
- Intermediate results saved periodically

### Performance
- GPU acceleration via CUDA (when available)
- Configurable tile size and overlap for different hardware
- Progress tracking with tqdm

### Data Integrity
- Automatic coordinate system detection
- Validation of input GeoJSON files
- Error handling with emergency checkpoints

## License

This project is provided as-is for plastic waste detection research and monitoring applications.

## Authors

PlastiScan v2 — Multi-Survey Drone Waste Detection System
