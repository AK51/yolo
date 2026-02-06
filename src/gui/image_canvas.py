"""Interactive image canvas for bounding box and polygon labeling"""
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QFont, QBrush, QPolygonF


class ImageCanvas(QLabel):
    """Interactive canvas for drawing bounding boxes and polygons on images"""
    
    bbox_created = pyqtSignal(list)  # Signal when bbox is created
    bbox_deleted = pyqtSignal(int)   # Signal when bbox is deleted
    polygon_created = pyqtSignal(list)  # Signal when polygon is created
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QLabel {
                background-color: #0d1117;
                border: 2px solid #00ff88;
                border-radius: 5px;
            }
        """)
        
        # Image data
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Drawing mode: 'bbox' or 'polygon'
        self.drawing_mode = 'bbox'
        
        # Bounding boxes (normalized coordinates: class_id, x_center, y_center, width, height)
        self.bboxes = []
        
        # Polygons (normalized coordinates: class_id, [x1, y1, x2, y2, ...])
        self.polygons = []
        
        # Drawing state for bounding boxes
        self.first_point = None
        self.current_point = None
        self.drawing = False
        self.hovered_bbox = -1
        
        # Drawing state for polygons
        self.polygon_points = []  # List of normalized points for current polygon
        self.drawing_polygon = False
        self.hovered_polygon = -1
        
        # Class info
        self.current_class_id = 0
        self.class_names = ["object"]
        self.show_class_names = True
        
        # Colors for bounding boxes and polygons
        self.bbox_colors = [
            QColor(0, 255, 136),    # Neon green
            QColor(255, 100, 100),  # Red
            QColor(100, 150, 255),  # Blue
            QColor(255, 200, 100),  # Orange
            QColor(200, 100, 255),  # Purple
        ]
    
    def set_drawing_mode(self, mode):
        """Set drawing mode: 'bbox' or 'polygon'"""
        self.drawing_mode = mode
        # Cancel any ongoing drawing
        self.first_point = None
        self.current_point = None
        self.drawing = False
        self.polygon_points = []
        self.drawing_polygon = False
        self.update()
    
    def load_image(self, image_path):
        """Load image from path"""
        self.original_pixmap = QPixmap(str(image_path))
        if self.original_pixmap.isNull():
            return False
        
        self.bboxes = []
        self.polygons = []
        self.first_point = None
        self.current_point = None
        self.drawing = False
        self.hovered_bbox = -1
        self.polygon_points = []
        self.drawing_polygon = False
        self.hovered_polygon = -1
        
        self._update_scaled_pixmap()
        self.update()  # Force repaint
        return True
    
    def _update_scaled_pixmap(self):
        """Update scaled pixmap to fit widget"""
        if not self.original_pixmap:
            return
        
        # Calculate scaling to fit widget while maintaining aspect ratio
        widget_width = self.width()
        widget_height = self.height()
        pixmap_width = self.original_pixmap.width()
        pixmap_height = self.original_pixmap.height()
        
        scale_w = widget_width / pixmap_width
        scale_h = widget_height / pixmap_height
        self.scale_factor = min(scale_w, scale_h) * 0.95  # 95% to leave margin
        
        scaled_width = int(pixmap_width * self.scale_factor)
        scaled_height = int(pixmap_height * self.scale_factor)
        
        self.scaled_pixmap = self.original_pixmap.scaled(
            scaled_width, scaled_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # Calculate offset to center image
        self.offset_x = (widget_width - scaled_width) // 2
        self.offset_y = (widget_height - scaled_height) // 2
        
        self.update()
    
    def set_bboxes(self, bboxes):
        """Set bounding boxes (YOLO format)"""
        self.bboxes = [list(bbox) for bbox in bboxes]
        self.update()
    
    def get_bboxes(self):
        """Get current bounding boxes"""
        return self.bboxes
    
    def get_polygons(self):
        """Get current polygons"""
        return self.polygons
    
    def clear_bboxes(self):
        """Clear all bounding boxes"""
        self.bboxes = []
        self.update()
    
    def clear_polygons(self):
        """Clear all polygons"""
        self.polygons = []
        self.update()
    
    def clear_all(self):
        """Clear all annotations (bboxes and polygons)"""
        self.bboxes = []
        self.polygons = []
        self.update()
    
    def undo_last_annotation(self):
        """Undo the last drawn annotation or point"""
        # If currently drawing a polygon, remove the last point
        if self.drawing_polygon and len(self.polygon_points) > 0:
            removed_point = self.polygon_points.pop()
            self.update()
            return True, 'point'
        
        # Otherwise, undo completed annotations
        # Undo polygon first if it exists, otherwise undo bbox
        if self.polygons:
            removed_polygon = self.polygons.pop()
            self.update()
            return True, 'polygon'
        elif self.bboxes:
            removed_bbox = self.bboxes.pop()
            self.update()
            return True, 'bbox'
        return False, None
    
    def undo_last_bbox(self):
        """Undo the last drawn bounding box (for backward compatibility)"""
        if self.bboxes:
            removed_bbox = self.bboxes.pop()
            self.update()
            return True
        return False
    
    def set_class_names(self, names):
        """Set class names"""
        self.class_names = names
        self.update()
    
    def set_current_class(self, class_id):
        """Set current class for drawing"""
        self.current_class_id = class_id
    
    def toggle_class_names(self):
        """Toggle display of class names"""
        self.show_class_names = not self.show_class_names
        self.update()
    
    def _screen_to_image_coords(self, screen_x, screen_y):
        """Convert screen coordinates to image coordinates (normalized 0-1)"""
        if not self.scaled_pixmap:
            return None, None
        
        # Adjust for offset
        img_x = screen_x - self.offset_x
        img_y = screen_y - self.offset_y
        
        # Check if within image bounds
        if img_x < 0 or img_y < 0 or img_x >= self.scaled_pixmap.width() or img_y >= self.scaled_pixmap.height():
            return None, None
        
        # Normalize to 0-1
        norm_x = img_x / self.scaled_pixmap.width()
        norm_y = img_y / self.scaled_pixmap.height()
        
        return norm_x, norm_y
    
    def _normalized_to_screen_rect(self, bbox):
        """Convert normalized bbox to screen rectangle"""
        if not self.scaled_pixmap:
            return None
        
        # bbox format: [class_id, x_center, y_center, width, height]
        x_center, y_center, width, height = bbox[1], bbox[2], bbox[3], bbox[4]
        
        # Convert to pixel coordinates
        img_width = self.scaled_pixmap.width()
        img_height = self.scaled_pixmap.height()
        
        center_x = x_center * img_width + self.offset_x
        center_y = y_center * img_height + self.offset_y
        box_width = width * img_width
        box_height = height * img_height
        
        x1 = int(center_x - box_width / 2)
        y1 = int(center_y - box_height / 2)
        
        return QRect(x1, y1, int(box_width), int(box_height))
    
    def _find_bbox_at_point(self, x, y):
        """Find bbox index at given point"""
        for i in range(len(self.bboxes) - 1, -1, -1):  # Check from top to bottom
            rect = self._normalized_to_screen_rect(self.bboxes[i])
            if rect and rect.contains(x, y):
                return i
        return -1
    
    def mousePressEvent(self, event):
        """Handle mouse press"""
        if not self.original_pixmap:
            return
        
        x, y = event.x(), event.y()
        
        # Right click - delete annotation or finish polygon
        if event.button() == Qt.RightButton:
            if self.drawing_mode == 'polygon' and self.drawing_polygon and len(self.polygon_points) >= 3:
                # Finish polygon on right-click
                self._finish_polygon()
                return
            
            # Delete bbox or polygon
            bbox_idx = self._find_bbox_at_point(x, y)
            if bbox_idx >= 0:
                del self.bboxes[bbox_idx]
                self.bbox_deleted.emit(bbox_idx)
                self.update()
            return
        
        # Left click - create bbox or add polygon point
        if event.button() == Qt.LeftButton:
            norm_x, norm_y = self._screen_to_image_coords(x, y)
            if norm_x is None:
                return
            
            if self.drawing_mode == 'bbox':
                # Bounding box mode
                if not self.drawing:
                    # First click - start drawing
                    self.first_point = (norm_x, norm_y)
                    self.drawing = True
                else:
                    # Second click - finish drawing
                    if self.first_point:
                        x1, y1 = self.first_point
                        x2, y2 = norm_x, norm_y
                        
                        # Calculate center and size
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        width = abs(x2 - x1)
                        height = abs(y2 - y1)
                        
                        # Only create if box has size
                        if width > 0.01 and height > 0.01:
                            new_bbox = [self.current_class_id, x_center, y_center, width, height]
                            self.bboxes.append(new_bbox)
                            self.bbox_created.emit(new_bbox)
                    
                    self.first_point = None
                    self.current_point = None
                    self.drawing = False
                    self.update()
                    
            elif self.drawing_mode == 'polygon':
                # Polygon mode - add point
                # Check if clicking near first point to close polygon
                if len(self.polygon_points) >= 3:
                    first_x, first_y = self.polygon_points[0]
                    distance = ((norm_x - first_x)**2 + (norm_y - first_y)**2)**0.5
                    if distance < 0.02:  # Close threshold
                        self._finish_polygon()
                        return
                
                # Add new point
                self.polygon_points.append((norm_x, norm_y))
                self.drawing_polygon = True
                self.update()
    
    def _finish_polygon(self):
        """Finish drawing the current polygon"""
        if len(self.polygon_points) >= 3:
            # Flatten points: [x1, y1, x2, y2, ...]
            flat_points = []
            for px, py in self.polygon_points:
                flat_points.extend([px, py])
            
            new_polygon = [self.current_class_id] + flat_points
            self.polygons.append(new_polygon)
            self.polygon_created.emit(new_polygon)
        
        self.polygon_points = []
        self.drawing_polygon = False
        self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move"""
        if not self.original_pixmap:
            return
        
        x, y = event.x(), event.y()
        
        # Update hovered bbox
        old_hovered = self.hovered_bbox
        self.hovered_bbox = self._find_bbox_at_point(x, y)
        if old_hovered != self.hovered_bbox:
            self.update()
        
        # Update current point while drawing bbox
        if self.drawing_mode == 'bbox' and self.drawing and self.first_point:
            norm_x, norm_y = self._screen_to_image_coords(x, y)
            if norm_x is not None:
                self.current_point = (norm_x, norm_y)
                self.update()
    
    def keyPressEvent(self, event):
        """Handle key press"""
        # Escape - cancel current drawing
        if event.key() == Qt.Key_Escape:
            if self.drawing_mode == 'bbox':
                self.first_point = None
                self.current_point = None
                self.drawing = False
            elif self.drawing_mode == 'polygon':
                self.polygon_points = []
                self.drawing_polygon = False
            self.update()
    
    def paintEvent(self, event):
        """Paint the canvas"""
        super().paintEvent(event)
        
        if not self.scaled_pixmap:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw image
        painter.drawPixmap(self.offset_x, self.offset_y, self.scaled_pixmap)
        
        # Draw existing bounding boxes
        for i, bbox in enumerate(self.bboxes):
            rect = self._normalized_to_screen_rect(bbox)
            if not rect:
                continue
            
            class_id = int(bbox[0])
            color = self.bbox_colors[class_id % len(self.bbox_colors)]
            
            # Highlight hovered box
            if i == self.hovered_bbox:
                pen = QPen(color, 3, Qt.SolidLine)
                painter.setPen(pen)
                painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 30)))
            else:
                pen = QPen(color, 2, Qt.SolidLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
            
            painter.drawRect(rect)
            
            # Draw class name
            if self.show_class_names and class_id < len(self.class_names):
                painter.setFont(QFont("Arial", 10, QFont.Bold))
                text_rect = QRect(rect.x(), rect.y() - 20, 100, 20)
                painter.fillRect(text_rect, color)
                painter.setPen(QPen(QColor(26, 26, 46)))
                painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, 
                               f" {self.class_names[class_id]}")
        
        # Draw existing polygons
        for i, polygon in enumerate(self.polygons):
            if len(polygon) < 7:  # Need at least class_id + 3 points (6 coords)
                continue
            
            class_id = int(polygon[0])
            color = self.bbox_colors[class_id % len(self.bbox_colors)]
            
            # Convert normalized coordinates to screen coordinates
            img_width = self.scaled_pixmap.width()
            img_height = self.scaled_pixmap.height()
            
            points = []
            for j in range(1, len(polygon), 2):
                if j + 1 < len(polygon):
                    norm_x = polygon[j]
                    norm_y = polygon[j + 1]
                    screen_x = norm_x * img_width + self.offset_x
                    screen_y = norm_y * img_height + self.offset_y
                    points.append(QPointF(screen_x, screen_y))
            
            if len(points) >= 3:
                qpolygon = QPolygonF(points)
                
                # Draw filled polygon with transparency
                pen = QPen(color, 2, Qt.SolidLine)
                painter.setPen(pen)
                painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
                painter.drawPolygon(qpolygon)
                
                # Draw class name at first point
                if self.show_class_names and class_id < len(self.class_names) and points:
                    painter.setFont(QFont("Arial", 10, QFont.Bold))
                    first_point = points[0]
                    text_rect = QRect(int(first_point.x()), int(first_point.y()) - 20, 100, 20)
                    painter.fillRect(text_rect, color)
                    painter.setPen(QPen(QColor(26, 26, 46)))
                    painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, 
                                   f" {self.class_names[class_id]}")
        
        # Draw current polygon being drawn
        if self.drawing_polygon and len(self.polygon_points) > 0:
            color = self.bbox_colors[self.current_class_id % len(self.bbox_colors)]
            img_width = self.scaled_pixmap.width()
            img_height = self.scaled_pixmap.height()
            
            # Draw lines between points
            pen = QPen(color, 2, Qt.SolidLine)
            painter.setPen(pen)
            
            for i in range(len(self.polygon_points)):
                x1, y1 = self.polygon_points[i]
                screen_x1 = x1 * img_width + self.offset_x
                screen_y1 = y1 * img_height + self.offset_y
                
                # Draw point
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QPointF(screen_x1, screen_y1), 4, 4)
                
                # Draw line to next point
                if i < len(self.polygon_points) - 1:
                    x2, y2 = self.polygon_points[i + 1]
                    screen_x2 = x2 * img_width + self.offset_x
                    screen_y2 = y2 * img_height + self.offset_y
                    painter.drawLine(QPointF(screen_x1, screen_y1), QPointF(screen_x2, screen_y2))
            
            # Draw line from last point to first point if we have at least 3 points
            if len(self.polygon_points) >= 3:
                x1, y1 = self.polygon_points[-1]
                x2, y2 = self.polygon_points[0]
                screen_x1 = x1 * img_width + self.offset_x
                screen_y1 = y1 * img_height + self.offset_y
                screen_x2 = x2 * img_width + self.offset_x
                screen_y2 = y2 * img_height + self.offset_y
                pen_dash = QPen(color, 2, Qt.DashLine)
                painter.setPen(pen_dash)
                painter.drawLine(QPointF(screen_x1, screen_y1), QPointF(screen_x2, screen_y2))
        
        # Draw current drawing box
        if self.drawing_mode == 'bbox' and self.drawing and self.first_point and self.current_point:
            x1, y1 = self.first_point
            x2, y2 = self.current_point
            
            # Convert to screen coordinates
            img_width = self.scaled_pixmap.width()
            img_height = self.scaled_pixmap.height()
            
            screen_x1 = int(x1 * img_width + self.offset_x)
            screen_y1 = int(y1 * img_height + self.offset_y)
            screen_x2 = int(x2 * img_width + self.offset_x)
            screen_y2 = int(y2 * img_height + self.offset_y)
            
            # Draw temporary box
            color = self.bbox_colors[self.current_class_id % len(self.bbox_colors)]
            pen = QPen(color, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 20)))
            
            rect = QRect(QPoint(screen_x1, screen_y1), QPoint(screen_x2, screen_y2))
            painter.drawRect(rect.normalized())
    
    def resizeEvent(self, event):
        """Handle resize"""
        super().resizeEvent(event)
        if self.original_pixmap:
            self._update_scaled_pixmap()
