from data.dataset_coco import DatasetCOCO
from data.dataset_coco_custom import DatasetCOCOCustom
from data.dataset_pascal import DatasetPASCAL
from data.dataset_pascal_custom import DatasetPASCALCustom


dataset_builder = {
    "coco": DatasetCOCO,
    "coco-custom-object-size": DatasetCOCOCustom,
    "pascal": DatasetPASCAL,
    "pascal-custom-object-size": DatasetPASCALCustom,
}
