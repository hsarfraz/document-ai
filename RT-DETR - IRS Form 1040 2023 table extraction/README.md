# RT-DETR to extract tables from IRS 1040 2023 forms

For IRS from 1040 document data parsing, I have previously uploaded a trained Donut model that is based on vision transformers. The donut model can perform single-shot parsing of 1040 forms and return parsed form values in json format. Vision transformers are cutting edge AI models, they still have some limitations when performing OCR related tasks, where they sometimes hallucinate. Secondly, they do not provide confidence level for extracted fields data, this makes it extremely challenging in making downstream decisions on when to accept a particular field value or drop the parsed value.

Especially when dealing with financial data, like Form 1040, accuracy and confidence values are of utmost importance. 

This article provides a working example of using multiple AI models to perform OCR of the form 1040 and extract text values in json format with confidence levels for each field. 




```bash
 -----------------------
| Classification Model  |         (Model is used to classify IRS Form 1040 by page)
 ----------------------- 
         |
         |
         |
 -----------------------
|       RT-DETR         | 
| Object Detection Model|         (Model trained to extract header and tables from Form 1040)
 -----------------------
         |
         |
         |
 -----------------------
|   Table Transformer   | 
|   Text OCR            |  
 -----------------------
```

## Classes for form 1040
The RT-DETR model is finetuned with 6 classes related to 1040 2023 form.
  ### Page 1 classes
  1040_pg1_header      - represents header of the page 1
  1040_pg1_tax_tbl     - represents table with financial values
  1040_pg1_sch_b       - represents table with schedule b financial values
  ### Page 2 classes
  1040_pg2_tax_tbl  
  1040_pg2_pay_tbl  
  1040_pg2_signature_frm  





[fake_1040_pg1](https://github.com/user-attachments/assets/847a6fa1-099c-47a3-af17-eda61dbd01f6)


```python
from ultralytics import RTDETR
import cv2
import supervision as sv 

# --------------------------
model_file = 'replace with path to model file /1014_2023_v1.pt'

# Load a trained model from local path
model = RTDETR(model_file)

# Display model information (optional)
model.info()

image_path = 'path to source image'

# read src image
img = cv2.imread(image_path)

# perform inference
results = model.predict(img, imgsz=1024) #imgsz is set to 1024 as the model is finetuned with this size

# use supervision library for parsing results and generating redline boxes
detections = sv.Detections.from_ultralytics(results[0])

#get bounding box and label annotator
bounding_box_annotator = sv.BoundingBoxAnnotator() 
label_annotaotr = sv.LabelAnnotator()

# generate labels for images
labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(detections['class_name'], detections.confidence)
]

# annotate images with labeled bounding boxes
annotated_image = bounding_box_annotator.annotate(
    scene=img.copy(),
    detections=detections
)
annotated_image = label_annotaotr.annotate(annotated_image, detections=detections, labels=labels)
# dummy counter for generated image names
count = 0
# write annotated image
cv2.imwrite('redlined_' + str(count) + '.png', annotated_image)

# crop bounding boxes and save 
for xyxy in detections.xyxy:
    cropped_image = sv.crop_image(image=img, xyxy=xyxy)
    count = count + 1
    cv2.imwrite('bboxes_' + str(count) + '.png', cropped_image)


```

