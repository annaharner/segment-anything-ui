import json
import os
import random
import nrrd
import SimpleITK as sitk
import glob

import cv2
import numpy as np
from PySide6.QtWidgets import QPushButton, QWidget, QFileDialog, QVBoxLayout, QLineEdit, QLabel, QCheckBox, QMessageBox

from segment_anything_ui.annotator import MasksAnnotation
from segment_anything_ui.config import Config


class FilesHolder:
    def __init__(self):
        self.files = []
        self.position = 0

    def add_files(self, files):
        self.files.extend(files)

    def get_next(self):
        self.position += 1
        if self.position >= len(self.files):
            self.position = 0
        return self.files[self.position]

    def get_previous(self):
        self.position -= 1
        if self.position < 0:
            self.position = len(self.files) - 1
        return self.files[self.position]


class SettingsLayout(QWidget):
    # Set the extensions for the mask and labels files
  #  MASK_EXTENSION = "_mask.seg.nrrd"
    LABELS_EXTENSION = "_labels.json"
    MASK_EXTENSION = "_mask.png"

    # Intialize the settings layout
    def __init__(self, parent, config, image_label) -> None:
        super().__init__(parent)
        self.config = config
        self.image_label = image_label
        self.actual_file: str = ""
        self.actual_shape = self.config.window_size
        self.layout = QVBoxLayout(self)
        self.open_files = QPushButton("Open Files")
        self.open_files.clicked.connect(self.on_open_files)
        self.apply_anno_to_next = QCheckBox("Apply current annotation to the next image")
        self.apply_anno_to_next.setChecked(False)
        self.apply_anno_to_next.clicked.connect(self.on_save_anno_prompt)
        self.save_auto_annos = QCheckBox("Save current image on next")
        self.save_auto_annos.setChecked(False)
        # self.save_auto_annos.clicked.connect(self.on_save_annos_masks)
        self.next_file = QPushButton(f"Next File [ {config.key_mapping.NEXT_FILE.name} ]")
        self.previous_file = QPushButton(f"Previous file [ {config.key_mapping.PREVIOUS_FILE.name} ]")
        self.previous_file.setShortcut(config.key_mapping.PREVIOUS_FILE.key)
        self.save_mask = QPushButton(f"Save Mask [ {config.key_mapping.SAVE_MASK.name} ]")
        self.save_mask.clicked.connect(self.on_save_mask)
        self.save_mask.setShortcut(config.key_mapping.SAVE_MASK.key)
        self.next_file.clicked.connect(self.on_next_file)
        self.next_file.setShortcut(config.key_mapping.NEXT_FILE.key)
        self.previous_file.clicked.connect(self.on_previous_file)
        self.checkpoint_path_label = QLabel(self, text="Checkpoint Path")
        self.checkpoint_path = QLineEdit(self, text=self.parent().config.default_weights)
        self.precompute_button = QPushButton("Precompute all embeddings")
        self.precompute_button.clicked.connect(self.on_precompute)
        self.delete_existing_annotation = QPushButton("Delete existing annotation")
        self.delete_existing_annotation.clicked.connect(self.on_delete_existing_annotation)
        self.Save_Seg_Stack = QPushButton("Save Segmentation Stack")
        self.Save_Seg_Stack.clicked.connect(self.on_save_seg_stack)
#       self.show_image = QPushButton("Show Image")
#       self.show_visualization = QPushButton("Show Visualization")
#        self.show_image.clicked.connect(self.on_show_image)
#        self.show_visualization.clicked.connect(self.on_show_visualization)
        self.show_text = QCheckBox("Show Text")
        self.show_text.clicked.connect(self.on_show_text)
        self.text_field_label = QLabel(self, text="Comma Separated Tags")
        self.tag_text_field = QLineEdit(self)
        self.tag_text_field.setPlaceholderText("Comma separated image tags")
        self.layout.addWidget(self.open_files)
        self.layout.addWidget(self.apply_anno_to_next)
        self.layout.addWidget(self.save_auto_annos)
        self.layout.addWidget(self.next_file)
        self.layout.addWidget(self.previous_file)
        self.layout.addWidget(self.save_mask)
        self.layout.addWidget(self.delete_existing_annotation)
        self.layout.addWidget(self.Save_Seg_Stack)
        self.layout.addWidget(self.show_text)
        self.layout.addWidget(self.text_field_label)
        self.layout.addWidget(self.tag_text_field)
        self.layout.addWidget(self.checkpoint_path_label)
        self.layout.addWidget(self.checkpoint_path)
        self.checkpoint_path.returnPressed.connect(self.on_checkpoint_path_changed)
        self.checkpoint_path.editingFinished.connect(self.on_checkpoint_path_changed)
        self.layout.addWidget(self.precompute_button)
 #       self.layout.addWidget(self.show_image)
 #       self.layout.addWidget(self.show_visualization)
        self.files = FilesHolder()
        self.neg_pnts = []
        self.pos_pnts = []
        self.bounding_box = []

    # Delete annotation files if they exist and user wants
    def on_delete_existing_annotation(self):
        path = os.path.split(self.actual_file)[0]
        basename = os.path.splitext(os.path.basename(self.actual_file))[0]
        mask_path = os.path.join(path, basename + self.MASK_EXTENSION)
      #  mask_path_png = os.path.join(path, basename + self.MASK_EXTENSION_PNG) # Added
        labels_path = os.path.join(path, basename + self.LABELS_EXTENSION)
        
        # Check and delete the mask file if it exists
        if os.path.exists(mask_path):
            os.remove(mask_path) 
            print(f"Deleted mask file: {mask_path}")
        
        # Check and delete the labels file if it exists
        if os.path.exists(labels_path):
            os.remove(labels_path)
            print(f"Deleted labels file: {labels_path}")

    def is_show_text(self):
        return self.show_text.isChecked()

    def on_show_text(self): 
        self.parent().update(self.parent().annotator.merge_image_visualization())

    # Save the promp points for the current annotation
    def on_save_anno_prompt(self):
        if self.apply_anno_to_next.isChecked():
            self.pos_pnts = self.image_label.positive_points
            self.neg_pnts = self.image_label.negative_points
            self.bounding_box = self.image_label.bounding_box

            print("Annotation will be applied to the next image!", len(self.pos_pnts), len(self.neg_pnts))
        else:
            print("Annotation will not be applied to the next image!")

    # Apply the saved prompt points to the next image
    def on_apply_anno_to_next(self):
        # First save annotations
        print("Applying annotation to the next image")
        self.image_label.positive_points = self.pos_pnts
        self.image_label.negative_points = self.neg_pnts
        self.image_label.bounding_box = self.bounding_box

        self.image_label.parent().annotator.make_prediction(self.image_label.get_annotations())
        self.image_label.parent().annotator.visualize_last_mask()

    # Save the masks and annotations if the user wants
    def on_save_annos_masks(self):
         from segment_anything_ui.annotation_layout import AnnotationLayout
         annotation_layout = self.parent().annotation_layout
         annotation_layout.on_save_annotation()
         print("Annotated")
         self.on_save_mask()

    def on_next_file(self):
        if self.save_auto_annos.isChecked():
            self.on_save_annos_masks()
        file = self.files.get_next()
        self._load_image(file)
        self.parent().setWindowTitle("Segment Anything UI - " + file)
        # Check if annotations should be propagated to the next image
        if self.apply_anno_to_next.isChecked():
            self.on_apply_anno_to_next()

    def on_previous_file(self):
        file = self.files.get_previous()
        self._load_image(file)
        self.parent().setWindowTitle("Segment Anything UI - " + file)
        # Check if annotations should be propagated to the pervious image
        if self.apply_anno_to_next.isChecked():
            self.on_apply_anno_to_next()

    # To visualize the image and annotation
    def _load_image(self, file: str, transfer=False):
        mask = file.split(".")[0] + self.MASK_EXTENSION
        labels = file.split(".")[0] + self.LABELS_EXTENSION
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        self.actual_shape = image.shape[:2][::-1]
        self.actual_file = file
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.dtype in [np.float32, np.float64, np.uint16]:
            image = (image / np.amax(image) * 255).astype("uint8")
        image = cv2.resize(image,
                           (int(self.parent().config.window_size[0]), self.parent().config.window_size[1]))
        self.parent().annotator.clear()
        self.parent().image_label.clear()
        self.parent().set_image(image)  
        if transfer and self.current_annotation:
            self._apply_annotation(self.current_annotation)
            self.parent().info_label.setText("Transferred annotation to the next image!")
        elif os.path.exists(mask) and os.path.exists(labels):
            self._load_annotation(mask, labels)
            self.parent().info_label.setText("Loaded annotation from saved files!")
            self.parent().update(self.parent().annotator.merge_image_visualization())
        else:
            self.parent().info_label.setText("No annotation found!")
         #   self.tag_text_field.setText("")
            self.parent().update(image)


    # # Load the annotation
    # def _load_annotation(self, mask, labels):
    #     mask = cv2.imread(mask, cv2.IMREAD_UNCHANGED)
    #     mask = cv2.resize(mask, (self.config.window_size[0], self.config.window_size[1]),
    #                       interpolation=cv2.INTER_NEAREST)
    #     with open(labels, "r") as fp:
    #         labels: dict[str, str] = json.load(fp)
    #         print(labels)
    #     masks_png = []
    #     new_labels = []
    #     if "instances" in labels:
    #         instance_labels = labels["instances"]
    #     else:
    #         instance_labels = labels

    #     if "tags" in labels:
    #         self.tag_text_field.setText(",".join(labels["tags"]))
    #     else:
    #         self.tag_text_field.setText("")
    #     for str_index, class_ in instance_labels.items():
    #         single_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    #         single_mask[mask == int(str_index)] = 255
    #         masks_png.append(single_mask)
    #         new_labels.append(class_)
    #     self.parent().annotator.masks = MasksAnnotation.from_masks(masks_png, new_labels)
    #     self.current_annotation = (masks_png, new_labels)

    def _load_annotation(self, mask, labels):
        mask = cv2.imread(mask, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (self.config.window_size[0], self.config.window_size[1]),
                        interpolation=cv2.INTER_NEAREST)
        
        with open(labels, "r") as fp:
            labels: dict[str, str] = json.load(fp)
            print(labels)

        masks_png = []
        new_labels = []

        if "instances" in labels:
            instance_labels = labels["instances"]
        else:
            instance_labels = labels

        if "tags" in labels:
            self.tag_text_field.setText(",".join(labels["tags"]))
        else:
            self.tag_text_field.setText("")

        for str_index, class_ in instance_labels.items():
            single_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
            single_mask[mask == int(str_index)] = 255
            masks_png.append(single_mask)
            new_labels.append(class_)

        self.parent().annotator.masks = MasksAnnotation.from_masks(masks_png, new_labels)
        self.current_annotation = (masks_png, new_labels)
        

    def _save_labels(self, labels, unique_labels):
        # Update the labels dictionary with unique labels
        labels["instances"] = {str(i): label for i, label in enumerate(unique_labels)}

        # Save the updated labels back to a file
        with open('updated_labels.json', 'w') as fp:  # Use a suitable file name
            json.dump(labels, fp)



 #   def on_show_image(self):
        pass

 #   def on_show_visualization(self):
        pass

    def on_precompute(self):
        pass
       
    def on_save_mask(self):
        path = os.path.split(self.actual_file)[0]
        tags = self.tag_text_field.text().split(",")
        tags = [tag.strip() for tag in tags]
        basename = os.path.splitext(os.path.basename(self.actual_file))[0]
        mask_path = os.path.join(path, basename + self.MASK_EXTENSION)
        labels_path = os.path.join(path, basename + self.LABELS_EXTENSION)
        masks = self.parent().get_mask()
        labels = {"instances": self.parent().get_labels(), "tags": tags}
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=4)
        masks = cv2.resize(masks, self.actual_shape, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(mask_path, masks)

 #   Save the stack of png masks into one nrrd file to visualize in 3D, ex. in slicer
    def on_save_seg_stack_2(self):
        path = os.path.split(self.actual_file)[0]
        print(path)
        files = glob.glob(os.path.join(path, '*mask.png'))
        files.sort()
        print(files)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        vol = reader.Execute()
        sitk.WriteImage(vol, 'seg_stack.nrrd')
    
    def on_save_seg_stack(self):
        """
        Save segmentation as a .seg.nrrd file compatible with 3D Slicer.
        
        Parameters:
            binary_label_map (numpy.ndarray): 3D array where each voxel is assigned a label.
            label_colors (dict): Dictionary mapping label values to RGB colors (e.g., {1: [0.5, 0.5, 0.5]}).
            output_path (str): Path to save the .seg.nrrd file.
        """
        
        # Write the image to .seg.nrrd format
        path = os.path.split(self.actual_file)[0]
        print(path)
        files = glob.glob(os.path.join(path, '*mask.png'))
        files.sort()
        print(files)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        sitk_image = reader.Execute()
        # Convert numpy array to SimpleITK Image
        # sitk_image = sitk.GetImageFromArray(self.parent().annotator.label_map)
        sitk_image.SetSpacing((1.0, 1.0, 1.0))  # Update as needed
        sitk_image.SetOrigin((0.0, 0.0, 0.0))   # Update as needed
        
        # Add segmentation-specific metadata
        num_segments = len(self.parent().annotation_layout.label_colors)
        sitk_image.SetMetaData("Segmentation_MasterRepresentation", "Binary labelmap")
        sitk_image.SetMetaData("Segmentation_ContainedRepresentationNames", "Binary labelmap")
        
        for idx, (label_value, color) in enumerate(self.parent().annotation_layout.label_colors.items()):
            # Set metadata for each segment
            sitk_image.SetMetaData(f"Segment{idx}_ID", f"Segment_{idx + 1}")
            sitk_image.SetMetaData(f"Segment{idx}_Name", f"Segment_{idx + 1}")
            sitk_image.SetMetaData(f"Segment{idx}_LabelValue", str(label_value))
            sitk_image.SetMetaData(f"Segment{idx}_Color", f"{color[0]} {color[1]} {color[2]}")
            sitk_image.SetMetaData(f"Segment{idx}_ColorAutoGenerated", "1")
            sitk_image.SetMetaData(f"Segment{idx}_NameAutoGenerated", "1")
            sitk_image.SetMetaData(f"Segment{idx}_Layer", "0")
            sitk_image.SetMetaData(
                f"Segment{idx}_Tags",
                "Segmentation.Status:inprogress|TerminologyEntry:Segmentation category and type - 3D Slicer General Anatomy list~SCT^85756007^Tissue~"
                "SCT^85756007^Tissue~^^~Anatomic codes - DICOM master list~^^~^^|"
            )
        
        # Write the image to .seg.nrrd format
        output_file = os.path.join(path, 'segmentation.seg.nrrd')
        sitk.WriteImage(sitk_image, output_file)  # Ensure the file name is included
        print(f"Saved segmentation to {output_file}")

        
    # def on_save_seg_stack(self):
    #     """
    #     Save segmentation as a .seg.nrrd file compatible with 3D Slicer.
    #     """
    #     import os
    #     import glob
    #     import SimpleITK as sitk
        
    #     # Get the directory path
    #     path = os.path.split(self.actual_file)[0]
    #     print(path)
        
    #     # Collect all mask files
    #     files = glob.glob(os.path.join(path, '*mask.png'))
    #     files.sort()
    #     print(files)
        
    #     # Load the mask series as a 3D image
    #     reader = sitk.ImageSeriesReader()
    #     reader.SetFileNames(files)
    #     sitk_image = reader.Execute()
        
    #     # Set spacing and origin (update these as per your reference image)
    #     sitk_image.SetSpacing((0.259, 0.259, 0.259))
    #     sitk_image.SetOrigin((0.0, 0.0, 0.0))
        
    #     # Add segmentation-specific metadata
    #     num_segments = len(self.parent().annotation_layout.label_colors)
    #     sitk_image.SetMetaData("Segmentation_MasterRepresentation", "Binary labelmap")
    #     sitk_image.SetMetaData("Segmentation_ContainedRepresentationNames", "Binary labelmap|Closed surface|")
    #     sitk_image.SetMetaData("Segmentation_ReferenceImageExtentOffset", "0 0 0")
    #     sitk_image.SetMetaData(
    #         "Segmentation_ConversionParameters",
    #         "Decimation factor|0.0|Desired reduction in the total number of polygons. Range: 0.0 (no decimation) to 1.0 (as much simplification as possible). "
    #         "Value of 0.8 typically reduces data set size by 80% without losing too much details.&Smoothing factor|0.5|Smoothing factor. "
    #         "Range: 0.0 (no smoothing) to 1.0 (strong smoothing)."
    #     )
        
    #     for idx, (label_value, color) in enumerate(self.parent().annotation_layout.label_colors.items()):
    #         # Set metadata for each segment
    #         sitk_image.SetMetaData(f"Segment{idx}_ID", f"Segment_{idx + 1}")
    #         sitk_image.SetMetaData(f"Segment{idx}_Name", f"Segment_{idx + 1}")
    #         sitk_image.SetMetaData(f"Segment{idx}_LabelValue", str(label_value))
    #         sitk_image.SetMetaData(f"Segment{idx}_Color", f"{color[0]} {color[1]} {color[2]}")
    #         sitk_image.SetMetaData(f"Segment{idx}_ColorAutoGenerated", "1")
    #         sitk_image.SetMetaData(f"Segment{idx}_NameAutoGenerated", "1")
    #         sitk_image.SetMetaData(f"Segment{idx}_Extent", "137 173 322 346 1939 2042")  # Replace with actual extents
    #         sitk_image.SetMetaData(f"Segment{idx}_Layer", "0")
    #         sitk_image.SetMetaData(
    #             f"Segment{idx}_Tags",
    #             "Segmentation.Status:inprogress|TerminologyEntry:Segmentation category and type - 3D Slicer General Anatomy list~SCT^85756007^Tissue~"
    #             "SCT^85756007^Tissue~^^~Anatomic codes - DICOM master list~^^~^^|"
    #         )
        
    #     # Write the image to a .seg.nrrd file
    #     output_file_path = os.path.join(path, 'segmentation.seg.nrrd')
    #     sitk.WriteImage(sitk_image, output_file_path)
    #     print(f"Saved segmentation to {output_file_path}")



    def on_checkpoint_path_changed(self):
        self.parent().sam = self.parent().init_sam()

    def on_open_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open Files", "", "Image Files (*.png *.jpg *.bmp *.tif *.tiff)")
        self.files.add_files(files)
        self.on_next_file()
