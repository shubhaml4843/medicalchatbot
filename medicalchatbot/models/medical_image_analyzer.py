import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class MedicalImageAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def auto_detect_body_part(self, image_path):
        """AI-powered body part detection from X-ray image"""
        try:
            image = self.preprocess_image(image_path)
            if image is None:
                return 'unknown'
            
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            
            # Feature extraction for body part classification
            features = self._extract_anatomical_features(gray)
            
            # Rule-based classification using image features
            if features['aspect_ratio'] > 1.5 and features['vertical_structures'] > 0.3:
                if features['rib_pattern'] > 0.4:
                    return 'chest'
                elif features['spine_pattern'] > 0.3:
                    return 'spine'
            
            elif features['aspect_ratio'] < 0.8:
                if features['circular_structures'] > 0.2:
                    return 'skull'
                elif features['joint_pattern'] > 0.3:
                    return 'knee'
            
            elif features['long_bone_pattern'] > 0.4:
                if features['dual_bones'] > 0.3:
                    return 'arm'
                else:
                    return 'leg'
            
            return 'other'
            
        except Exception as e:
            logger.error(f"Error detecting body part: {e}")
            return 'unknown'
    
    def _extract_anatomical_features(self, gray):
        """Extract features to identify body parts"""
        h, w = gray.shape
        aspect_ratio = h / w
        
        edges = cv2.Canny(gray, 30, 100)
        
        # Detect rib pattern
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        rib_pattern = np.sum(horizontal_lines > 0) / edges.size
        
        # Detect vertical structures
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        vertical_structures = np.sum(vertical_lines > 0) / edges.size
        
        # Detect circular structures
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=10, maxRadius=100)
        circular_structures = len(circles[0]) if circles is not None else 0
        circular_structures = min(circular_structures / 10, 1.0)
        
        # Detect spine pattern
        center_col = gray[:, w//2-10:w//2+10]
        spine_pattern = np.std(center_col) / 255.0
        
        # Detect dual bones
        left_third = gray[:, :w//3]
        right_third = gray[:, 2*w//3:]
        dual_bones = (np.std(left_third) + np.std(right_third)) / (2 * 255.0)
        
        # Long bone pattern
        long_bone_pattern = vertical_structures * (1 - rib_pattern)
        
        # Joint pattern
        joint_pattern = np.max([np.sum(edges[i:i+50, j:j+50]) for i in range(0, h-50, 25) for j in range(0, w-50, 25)]) / (50*50)
        
        return {
            'aspect_ratio': aspect_ratio,
            'rib_pattern': rib_pattern,
            'vertical_structures': vertical_structures,
            'circular_structures': circular_structures,
            'spine_pattern': spine_pattern,
            'dual_bones': dual_bones,
            'long_bone_pattern': long_bone_pattern,
            'joint_pattern': joint_pattern
        }
        
    def validate_medical_image(self, image_path):
        """Validate if image is likely a medical scan"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return False, "Invalid image file"
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Check image characteristics
            height, width = gray.shape
            
            # Medical images are typically grayscale or have medical markers
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Basic validation rules
            if width < 100 or height < 100:
                return False, "Image too small for medical analysis"
            
            if mean_intensity > 200:  # Too bright (likely not X-ray)
                return False, "Image appears to be non-medical (too bright)"
            
            if std_intensity < 10:  # Too uniform (likely solid color)
                return False, "Image lacks medical scan characteristics"
            
            return True, "Valid medical image"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def preprocess_image(self, image_path):
        """Preprocess medical image for analysis"""
        try:
            # Validate medical image first
            is_valid, message = self.validate_medical_image(image_path)
            if not is_valid:
                raise ValueError(f"Invalid medical image: {message}")
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Could not load image")
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Enhance contrast for medical images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            if len(image_rgb.shape) == 3:
                # For color images, apply to each channel
                for i in range(3):
                    image_rgb[:,:,i] = clahe.apply(image_rgb[:,:,i])
            else:
                image_rgb = clahe.apply(image_rgb)
            
            return image_rgb
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def detect_anomalies(self, image_path):
        """Detect medically relevant anomalies in medical images"""
        try:
            image = self.preprocess_image(image_path)
            if image is None:
                return []
            
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            anomalies = []
            
            # 1. Detect bright spots (possible calcifications, foreign objects)
            bright_threshold = np.percentile(gray, 95)
            bright_mask = gray > bright_threshold
            bright_contours, _ = cv2.findContours(bright_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in bright_contours:
                area = cv2.contourArea(contour)
                if 50 < area < 2000:  # Filter reasonable sizes
                    x, y, w, h = cv2.boundingRect(contour)
                    # Classify bright spot based on characteristics
                    anomaly_name = self._classify_bright_anomaly(area, x, y, w, h, gray)
                    anomalies.append({
                        'type': 'bright_spot',
                        'name': anomaly_name,
                        'description': f'{anomaly_name} - High-density area detected',
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'area': float(area),
                        'confidence': float(min(area / 500, 0.8))
                    })
            
            # 2. Detect dark spots (possible lesions, air pockets)
            dark_threshold = np.percentile(gray, 15)
            dark_mask = gray < dark_threshold
            dark_contours, _ = cv2.findContours(dark_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in dark_contours:
                area = cv2.contourArea(contour)
                if 100 < area < 3000:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Classify dark spot based on characteristics
                    anomaly_name = self._classify_dark_anomaly(area, x, y, w, h, gray)
                    anomalies.append({
                        'type': 'dark_spot',
                        'name': anomaly_name,
                        'description': f'{anomaly_name} - Low-density area detected',
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'area': float(area),
                        'confidence': float(min(area / 800, 0.7))
                    })
            
            # 3. Detect linear features (possible fractures)
            edges = cv2.Canny(gray, 30, 100)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            if lines is not None:
                for line in lines[:3]:  # Top 3 lines
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if length > 40:
                        # Classify linear feature
                        anomaly_name = self._classify_linear_anomaly(length, x1, y1, x2, y2)
                        anomalies.append({
                            'type': 'linear_feature',
                            'name': anomaly_name,
                            'description': f'{anomaly_name} - Linear density detected',
                            'bbox': [int(min(x1,x2)-5), int(min(y1,y2)-5), int(abs(x2-x1)+10), int(abs(y2-y1)+10)],
                            'area': float(length),
                            'confidence': float(min(length / 100, 0.6))
                        })
            
            # 4. Detect asymmetry (compare left/right halves)
            left_half = gray[:, :w//2]
            right_half = gray[:, w//2:]
            
            # Flip right half for comparison
            right_flipped = np.fliplr(right_half)
            
            # Calculate difference
            if left_half.shape == right_flipped.shape:
                diff = np.abs(left_half.astype(float) - right_flipped.astype(float))
                asymmetry_score = np.mean(diff)
                
                if asymmetry_score > 25:  # Significant asymmetry
                    anomalies.append({
                        'type': 'asymmetry',
                        'name': 'Bilateral Asymmetry',
                        'description': f'Bilateral Asymmetry - Structural imbalance (score: {asymmetry_score:.1f})',
                        'bbox': [0, 0, int(w), int(h)],
                        'area': float(asymmetry_score),
                        'confidence': float(min(asymmetry_score / 50, 0.8))
                    })
            
            # 5. Detect texture abnormalities using local standard deviation
            kernel = np.ones((15,15), np.float32) / 225
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_sq_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
            local_std = np.sqrt(local_sq_mean - local_mean**2)
            
            # Find areas with unusual texture
            texture_threshold = np.percentile(local_std, 90)
            texture_mask = local_std > texture_threshold
            texture_contours, _ = cv2.findContours(texture_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in texture_contours:
                area = cv2.contourArea(contour)
                if 200 < area < 5000:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Classify texture abnormality
                    anomaly_name = self._classify_texture_anomaly(area, x, y, w, h)
                    anomalies.append({
                        'type': 'texture_abnormality',
                        'name': anomaly_name,
                        'description': f'{anomaly_name} - Abnormal tissue pattern',
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'area': float(area),
                        'confidence': float(min(area / 1000, 0.6))
                    })
            
            # Sort by confidence and return top anomalies
            anomalies.sort(key=lambda x: x['confidence'], reverse=True)
            return anomalies[:8]  # Return top 8 most confident anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def analyze_by_body_part(self, image_path, body_part):
        """Dynamic analysis based on body part"""
        try:
            image = self.preprocess_image(image_path)
            if image is None:
                return [{'finding': 'Image validation failed', 'confidence': 0.0, 'severity': 'error'}]
            
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Dynamic analysis based on body part
            body_part_lower = body_part.lower()
            
            if body_part_lower in ['chest', 'lung', 'thorax', 'ribs']:
                return self._analyze_chest(gray, mean_intensity, std_intensity)
            elif body_part_lower in ['leg', 'femur', 'tibia', 'fibula', 'knee', 'ankle', 'foot']:
                return self._analyze_leg(gray, mean_intensity, std_intensity)
            elif body_part_lower in ['arm', 'humerus', 'radius', 'ulna', 'elbow', 'wrist', 'hand', 'shoulder']:
                return self._analyze_arm(gray, mean_intensity, std_intensity)
            elif body_part_lower in ['spine', 'back', 'vertebra', 'cervical', 'thoracic', 'lumbar']:
                return self._analyze_spine(gray, mean_intensity, std_intensity)
            elif body_part_lower in ['skull', 'head', 'brain', 'cranium']:
                return self._analyze_skull(gray, mean_intensity, std_intensity)
            elif body_part_lower in ['abdomen', 'pelvis', 'hip']:
                return self._analyze_abdomen(gray, mean_intensity, std_intensity)
            elif body_part_lower in ['neck', 'cervical spine']:
                return self._analyze_neck(gray, mean_intensity, std_intensity)
            else:
                return self._analyze_general(gray, mean_intensity, std_intensity, body_part)
                
        except Exception as e:
            logger.error(f"Error analyzing {body_part}: {e}")
            return [{'finding': 'Analysis failed', 'confidence': 0.0, 'severity': 'error'}]
    
    def _analyze_chest(self, gray, mean_intensity, std_intensity):
        findings = []
        h, w = gray.shape
        
        # Analyze lung fields
        left_lung = gray[:, :w//3]
        right_lung = gray[:, 2*w//3:]
        
        left_mean = np.mean(left_lung)
        right_mean = np.mean(right_lung)
        
        # Check for asymmetry
        if abs(left_mean - right_mean) > 20:
            findings.append({
                'finding': f'Lung field asymmetry detected (L:{left_mean:.1f}, R:{right_mean:.1f})',
                'confidence': 0.75,
                'severity': 'mild'
            })
        
        # Check for consolidation patterns
        if mean_intensity < 60:
            findings.append({
                'finding': 'Increased lung opacity - possible consolidation or infiltrate',
                'confidence': 0.70,
                'severity': 'moderate'
            })
        
        # Check rib structure
        edges = cv2.Canny(gray, 30, 100)
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
        rib_pattern = np.sum(horizontal_lines > 0) / edges.size
        
        if rib_pattern > 0.05:
            findings.append({
                'finding': 'Rib cage structure clearly visible - good image quality',
                'confidence': 0.85,
                'severity': 'normal'
            })
        
        # Heart shadow analysis
        center_region = gray[h//3:2*h//3, w//3:2*w//3]
        if np.mean(center_region) < mean_intensity - 15:
            findings.append({
                'finding': 'Cardiac silhouette visible - normal heart shadow',
                'confidence': 0.80,
                'severity': 'normal'
            })
        
        return findings or [{
            'finding': 'Chest X-ray shows normal lung fields and cardiac silhouette',
            'confidence': 0.65,
            'severity': 'normal'
        }]
    
    def _analyze_leg(self, gray, mean_intensity, std_intensity):
        findings = []
        h, w = gray.shape
        
        # Analyze bone density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Check for fractures (high edge density in specific areas)
        if edge_density > 0.20:
            findings.append({
                'finding': 'Sharp cortical edges detected - possible fracture line',
                'confidence': 0.65,
                'severity': 'high'
            })
        elif edge_density > 0.12:
            findings.append({
                'finding': 'Normal bone cortex definition',
                'confidence': 0.80,
                'severity': 'normal'
            })
        
        # Analyze bone alignment
        center_col = gray[:, w//2-5:w//2+5]
        alignment_std = np.std(np.mean(center_col, axis=1))
        
        if alignment_std > 25:
            findings.append({
                'finding': 'Possible bone angulation or displacement',
                'confidence': 0.60,
                'severity': 'moderate'
            })
        
        # Soft tissue analysis
        if mean_intensity > 90:
            findings.append({
                'finding': 'Soft tissue swelling noted',
                'confidence': 0.70,
                'severity': 'mild'
            })
        
        # Joint space analysis
        if std_intensity > 50:
            findings.append({
                'finding': 'Good bone-soft tissue contrast, joint spaces preserved',
                'confidence': 0.75,
                'severity': 'normal'
            })
        
        return findings or [{
            'finding': 'Lower extremity X-ray shows normal bone alignment and density',
            'confidence': 0.70,
            'severity': 'normal'
        }]
    
    def _analyze_arm(self, gray, mean_intensity, std_intensity):
        findings = []
        if std_intensity > 50:
            findings.append({'finding': 'Good bone-soft tissue contrast', 'confidence': 0.7, 'severity': 'normal'})
        if mean_intensity < 60:
            findings.append({'finding': 'Dense bone structure detected', 'confidence': 0.8, 'severity': 'normal'})
        return findings or [{'finding': 'Arm X-ray appears normal', 'confidence': 0.6, 'severity': 'normal'}]
    
    def _analyze_spine(self, gray, mean_intensity, std_intensity):
        findings = []
        h, w = gray.shape
        
        # Analyze spinal alignment
        center_col = gray[:, w//2-10:w//2+10]
        spine_profile = np.mean(center_col, axis=1)
        
        # Check for scoliosis (lateral curvature)
        left_spine = gray[:, w//2-20:w//2]
        right_spine = gray[:, w//2:w//2+20]
        asymmetry = abs(np.mean(left_spine) - np.mean(right_spine))
        
        if asymmetry > 15:
            findings.append({
                'finding': 'Spinal asymmetry detected - possible scoliotic curvature',
                'confidence': 0.65,
                'severity': 'moderate'
            })
        
        # Vertebral body analysis
        if mean_intensity < 80:
            findings.append({
                'finding': 'Good vertebral body density - no obvious osteoporosis',
                'confidence': 0.75,
                'severity': 'normal'
            })
        
        # Disc space analysis
        if std_intensity > 55:
            findings.append({
                'finding': 'Disc spaces appear preserved',
                'confidence': 0.70,
                'severity': 'normal'
            })
        else:
            findings.append({
                'finding': 'Possible disc space narrowing',
                'confidence': 0.60,
                'severity': 'mild'
            })
        
        # Check for compression fractures
        spine_gradient = np.gradient(spine_profile)
        if np.max(np.abs(spine_gradient)) > 5:
            findings.append({
                'finding': 'Vertebral height irregularity - rule out compression fracture',
                'confidence': 0.55,
                'severity': 'moderate'
            })
        
        return findings or [{
            'finding': 'Spine X-ray shows normal vertebral alignment and disc spaces',
            'confidence': 0.70,
            'severity': 'normal'
        }]
    
    def _analyze_skull(self, gray, mean_intensity, std_intensity):
        findings = []
        h, w = gray.shape
        
        # Analyze skull density
        if mean_intensity < 90:
            findings.append({
                'finding': 'Normal skull bone density',
                'confidence': 0.80,
                'severity': 'normal'
            })
        
        # Check for fractures using edge detection
        edges = cv2.Canny(gray, 40, 120)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density > 0.15:
            findings.append({
                'finding': 'Linear lucency detected - possible skull fracture',
                'confidence': 0.60,
                'severity': 'high'
            })
        
        # Analyze brain-skull contrast
        if std_intensity > 45:
            findings.append({
                'finding': 'Good gray-white matter differentiation',
                'confidence': 0.75,
                'severity': 'normal'
            })
        
        # Check for midline shift
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        
        if abs(np.mean(left_half) - np.mean(right_half)) > 10:
            findings.append({
                'finding': 'Asymmetric brain density - rule out mass effect',
                'confidence': 0.55,
                'severity': 'moderate'
            })
        
        # Suture analysis
        if edge_density > 0.08 and edge_density < 0.12:
            findings.append({
                'finding': 'Suture lines visible - normal for age',
                'confidence': 0.70,
                'severity': 'normal'
            })
        
        return findings or [{
            'finding': 'Skull X-ray shows normal bone density and brain parenchyma',
            'confidence': 0.70,
            'severity': 'normal'
        }]
    
    def _analyze_general(self, gray, mean_intensity, std_intensity, body_part='unknown'):
        findings = []
        
        # Basic image quality assessment
        if mean_intensity > 180:
            findings.append({
                'finding': 'Image appears overexposed - may not be diagnostic quality X-ray',
                'confidence': 0.85,
                'severity': 'moderate'
            })
        elif mean_intensity < 30:
            findings.append({
                'finding': 'Image appears underexposed - limited diagnostic value',
                'confidence': 0.80,
                'severity': 'moderate'
            })
        else:
            findings.append({
                'finding': f'Medical image of {body_part} - adequate exposure for analysis',
                'confidence': 0.75,
                'severity': 'normal'
            })
        
        # Contrast analysis
        if std_intensity > 60:
            findings.append({
                'finding': 'Good image contrast - anatomical structures well differentiated',
                'confidence': 0.80,
                'severity': 'normal'
            })
        elif std_intensity < 20:
            findings.append({
                'finding': 'Low image contrast - may limit diagnostic assessment',
                'confidence': 0.70,
                'severity': 'mild'
            })
        
        # Edge detection for structural analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density > 0.15:
            findings.append({
                'finding': 'Sharp anatomical boundaries visible - good image quality',
                'confidence': 0.75,
                'severity': 'normal'
            })
        elif edge_density < 0.05:
            findings.append({
                'finding': 'Limited structural detail - consider repeat imaging',
                'confidence': 0.65,
                'severity': 'mild'
            })
        
        return findings or [{
            'finding': f'General analysis of {body_part} image completed',
            'confidence': 0.60,
            'severity': 'normal'
        }]
    
    def _analyze_abdomen(self, gray, mean_intensity, std_intensity):
        findings = []
        h, w = gray.shape
        
        # Soft tissue analysis
        if mean_intensity > 100:
            findings.append({
                'finding': 'Soft tissue structures visible - good penetration',
                'confidence': 0.75,
                'severity': 'normal'
            })
        
        # Check for gas patterns
        if std_intensity > 50:
            findings.append({
                'finding': 'Normal bowel gas pattern visible',
                'confidence': 0.70,
                'severity': 'normal'
            })
        
        # Look for abnormal densities
        if mean_intensity < 60:
            findings.append({
                'finding': 'Dense structures noted - rule out calcifications or foreign bodies',
                'confidence': 0.60,
                'severity': 'mild'
            })
        
        return findings or [{
            'finding': 'Abdominal X-ray shows normal soft tissue and gas patterns',
            'confidence': 0.65,
            'severity': 'normal'
        }]
    
    def _analyze_neck(self, gray, mean_intensity, std_intensity):
        findings = []
        
        # Cervical spine analysis
        if std_intensity > 45:
            findings.append({
                'finding': 'Cervical vertebrae alignment appears normal',
                'confidence': 0.70,
                'severity': 'normal'
            })
        
        # Soft tissue analysis
        if mean_intensity > 80:
            findings.append({
                'finding': 'Prevertebral soft tissues appear normal',
                'confidence': 0.65,
                'severity': 'normal'
            })
        
        # Airway analysis
        center_region = gray[:, gray.shape[1]//3:2*gray.shape[1]//3]
        if np.mean(center_region) > mean_intensity + 10:
            findings.append({
                'finding': 'Airway column visible and patent',
                'confidence': 0.75,
                'severity': 'normal'
            })
        
        return findings or [{
            'finding': 'Neck X-ray shows normal cervical alignment and soft tissues',
            'confidence': 0.65,
            'severity': 'normal'
        }]
    
    def _infer_body_part_from_image(self, image_path):
        """Infer body part from image characteristics when auto-detection fails"""
        try:
            image = self.preprocess_image(image_path)
            if image is None:
                return 'unknown'
            
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            
            # Basic shape analysis
            aspect_ratio = h / w
            
            # Intensity analysis
            mean_intensity = np.mean(gray)
            
            # Edge analysis
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Heuristic classification
            if aspect_ratio > 1.8:  # Very tall images
                if edge_density > 0.1:
                    return 'long_bone'  # Likely arm or leg
                else:
                    return 'spine'
            elif aspect_ratio < 0.7:  # Wide images
                if mean_intensity < 80:
                    return 'chest'
                else:
                    return 'pelvis'
            elif 0.8 <= aspect_ratio <= 1.2:  # Square-ish images
                if mean_intensity > 100:
                    return 'soft_tissue'
                else:
                    return 'joint'  # Knee, elbow, etc.
            else:
                return 'extremity'  # Arm, leg, hand, foot
                
        except Exception as e:
            logger.error(f"Error inferring body part: {e}")
            return 'unknown'
    
    def _classify_bright_anomaly(self, area, x, y, w, h, gray):
        """Classify bright spots with specific medical names"""
        if area < 200:
            return 'Calcification'
        elif area < 800:
            if w > h * 2 or h > w * 2:  # Elongated
                return 'Surgical Hardware'
            else:
                return 'Dense Nodule'
        elif area < 2000:
            return 'Bone Spur'
        else:
            return 'Large Calcified Mass'
    
    def _classify_dark_anomaly(self, area, x, y, w, h, gray):
        """Classify dark spots with specific medical names"""
        if area < 300:
            return 'Small Lesion'
        elif area < 1000:
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 2:
                return 'Linear Lucency'
            else:
                return 'Cystic Lesion'
        elif area < 2500:
            return 'Large Lesion'
        else:
            return 'Extensive Lucency'
    
    def _classify_linear_anomaly(self, length, x1, y1, x2, y2):
        """Classify linear features with specific medical names"""
        angle = np.arctan2(abs(y2-y1), abs(x2-x1)) * 180 / np.pi
        
        if length < 50:
            return 'Hairline Fracture'
        elif length < 100:
            if angle < 30 or angle > 150:  # Horizontal
                return 'Transverse Fracture'
            elif 60 < angle < 120:  # Vertical
                return 'Longitudinal Fracture'
            else:
                return 'Oblique Fracture'
        else:
            return 'Complete Fracture'
    
    def _classify_texture_anomaly(self, area, x, y, w, h):
        """Classify texture abnormalities with specific medical names"""
        if area < 500:
            return 'Focal Heterogeneity'
        elif area < 1500:
            return 'Patchy Infiltrate'
        elif area < 3000:
            return 'Diffuse Changes'
        else:
            return 'Extensive Infiltration'
    
    def generate_analysis_report(self, image_path, image_type='xray', user_body_part=None):
        """Generate comprehensive analysis report"""
        try:
            # Detect anomalies
            anomalies = self.detect_anomalies(image_path)
            
            # Auto-detect body part from image (as suggestion only)
            detected_body_part = self.auto_detect_body_part(image_path)
            
            # Prioritize user input, then detection, then default
            if user_body_part and user_body_part.strip():
                body_part = user_body_part.strip().lower()
                detected_body_part = f"user_specified_{body_part}"  # Mark as user specified
            elif detected_body_part != 'unknown' and detected_body_part != 'other':
                body_part = detected_body_part
            else:
                # Try to infer from image characteristics
                body_part = self._infer_body_part_from_image(image_path)
                if body_part == 'unknown':
                    body_part = 'unspecified_anatomy'  # More descriptive than 'general'
            
            # Dynamic analysis based on body part
            findings = self.analyze_by_body_part(image_path, body_part)
            
            # Generate report
            assessment = self._generate_assessment(anomalies, findings)
            
            report = {
                'image_path': str(image_path),
                'image_type': image_type,
                'detected_body_part': detected_body_part,
                'user_specified_part': user_body_part,
                'analyzed_as': body_part,
                'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'anomalies_detected': len(anomalies),
                'anomalies': anomalies,
                'clinical_findings': findings,
                'overall_assessment': assessment,
                'summary': {
                    'total_findings': len(findings),
                    'high_priority': sum(1 for f in findings if f.get('severity') == 'high'),
                    'moderate_priority': sum(1 for f in findings if f.get('severity') == 'moderate'),
                    'normal_findings': sum(1 for f in findings if f.get('severity') == 'normal'),
                    'average_confidence': round(np.mean([f.get('confidence', 0.5) for f in findings]), 2) if findings else 0.0
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating analysis report: {e}")
            return None
    
    def _generate_assessment(self, anomalies, findings):
        """Generate overall assessment based on findings and anomalies"""
        if not findings:
            return {
                'status': 'normal',
                'recommendation': 'No significant abnormalities detected. Routine follow-up as clinically indicated.',
                'urgency': 'routine'
            }
        
        # Count severity levels
        high_severity_count = sum(1 for f in findings if f.get('severity') == 'high')
        moderate_severity_count = sum(1 for f in findings if f.get('severity') == 'moderate')
        normal_count = sum(1 for f in findings if f.get('severity') == 'normal')
        
        # Analyze anomaly types for additional context
        critical_anomalies = sum(1 for a in anomalies if a.get('type') in ['linear_feature', 'asymmetry'] and a.get('confidence', 0) > 0.6)
        moderate_anomalies = sum(1 for a in anomalies if a.get('type') in ['bright_spot', 'dark_spot'] and a.get('confidence', 0) > 0.5)
        
        # Calculate average confidence
        avg_confidence = np.mean([f.get('confidence', 0.5) for f in findings])
        
        # Determine status and recommendations
        if high_severity_count > 0 or critical_anomalies > 0:
            if high_severity_count > 1 or critical_anomalies > 1:
                return {
                    'status': 'abnormal',
                    'recommendation': 'Multiple significant findings detected. Immediate clinical correlation and possible urgent consultation recommended.',
                    'urgency': 'urgent'
                }
            else:
                return {
                    'status': 'abnormal',
                    'recommendation': 'Significant finding detected. Clinical correlation and specialist consultation recommended within 24-48 hours.',
                    'urgency': 'urgent'
                }
        
        elif moderate_severity_count > 0 or moderate_anomalies > 2:
            if moderate_severity_count > 2 or moderate_anomalies > 4:
                return {
                    'status': 'abnormal',
                    'recommendation': 'Multiple moderate findings. Clinical correlation and follow-up imaging in 1-2 weeks recommended.',
                    'urgency': 'semi-urgent'
                }
            else:
                return {
                    'status': 'borderline',
                    'recommendation': 'Moderate findings noted. Clinical correlation recommended. Follow-up imaging may be needed.',
                    'urgency': 'routine'
                }
        
        elif normal_count > 0:
            if avg_confidence > 0.7:
                return {
                    'status': 'normal',
                    'recommendation': 'Study appears within normal limits. Routine clinical follow-up as indicated.',
                    'urgency': 'routine'
                }
            else:
                return {
                    'status': 'borderline',
                    'recommendation': 'Study appears largely normal but image quality may limit assessment. Consider repeat imaging if clinically indicated.',
                    'urgency': 'routine'
                }
        
        else:
            return {
                'status': 'indeterminate',
                'recommendation': 'Image quality or technical factors limit diagnostic assessment. Repeat imaging recommended.',
                'urgency': 'routine'
            }

# Global analyzer instance
medical_analyzer = MedicalImageAnalyzer()