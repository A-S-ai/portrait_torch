# import supervisely_lib as sly
from supervisely_lib.project.project import Project
project = Project("D:\py final\portrait-matting\supervisely-person-datasets", sly.OpenMode.READ)
#打印数据集相关信息
print("Project name: ", project.name)
print("Project directory: ", project.directory)
print("Total images: ", project.total_items)
print("Dataset names: ", project.datasets.keys())