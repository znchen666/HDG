# from collections import defaultdict
# file_path1 = r'/data0/czn/longtail_workspace/multi-domain-imbalance/data/STL10/img/stl10_imagelist/stl10_train_new.txt' # 文件路径
# file_path2 = r'/data0/czn/longtail_workspace/multi-domain-imbalance/data/Office31/office31/image_list/amazon.txt'
# file_path3 = r'/data0/czn/longtail_workspace/multi-domain-imbalance/data/Visda2017/train/visda_imagelist/visda_train_new.txt'
#
# class_dict = {}
# with open(file_path1, "r") as file:
#     for line in file:
#         line = line.strip()
#         class_name = line.split('/')[0]
#         class_label = int(line.split(' ')[1])
#         if class_label not in class_dict.keys():
#             class_dict[class_label] = class_name
#
# with open(file_path2, "r") as file:
#     for line in file:
#         line = line.strip()
#         class_name = line.split('/')[2]
#         class_label = int(line.split(' ')[1])
#         if class_label not in class_dict.keys():
#             class_dict[class_label] = class_name
#
# with open(file_path3, "r") as file:
#     for line in file:
#         line = line.strip()
#         class_name = line.split('/')[0]
#         class_label = int(line.split(' ')[1])
#         if class_label not in class_dict.keys():
#             class_dict[class_label] = class_name
# print(class_dict)
#
# sorted_values = [class_dict[key] for key in sorted(class_dict.keys())]
# print(sorted_values)


from collections import defaultdict
file_path1 = r'/data0/czn/longtail_workspace/multi-domain-imbalance/data/DomainNet/balance_image_list/Sketch_test.txt' # 文件路径

class_dict = {}
with open(file_path1, "r") as file:
    for line in file:
        class_name = line.split('/')[1]
        class_label = int(line.split(' ')[1])
        if class_label not in class_dict.keys():
            class_dict[class_label] = class_name

sorted_values = [class_dict[key] for key in sorted(class_dict.keys())]
print(sorted_values)
