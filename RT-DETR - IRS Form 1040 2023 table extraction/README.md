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

## Model classes for form 1040
The finetuned RT-DETR model has 6 classes that are named with page numbers to improve the detection