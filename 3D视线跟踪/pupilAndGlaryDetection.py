import cv2
import numpy as np
class Detection:
    def __init__(self,img,n):
        self.img = img
        self.n = n

    def read_net(self,config, model):
        net = cv2.dnn.readNetFromDarknet(config, model)  # 是 OpenCV 的深度神经网络（DNN）模块中的一个函数，用于加载使用 YOLO（You Only Look Once）框架训练的神经网络。
        return net

    def draw_prediction(self,class_id, confidence, left, top, right, bottom, frame):
        # Draw a rectangle around the detected object.confidence 是置信度
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        label = f"{confidence:.2f}"
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    def post_process(self,frame, outs):
        # process the outputs of YOLO;outs are the output of YOLO,include many detect results for the frame
        class_ids = []
        confidences = []
        boxes = []
        centers = []
        height, width = frame.shape[:2]
        # 遍历每个输出结果
        # detection 是一个包含物体位置信息和类别概率的列表
        global z
        z = 1
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)  # 找到置信度最高的类别
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    centers.append((center_x, center_y))

                    # Rectangle coordinates
                    # 使用 YOLO 输出中的相对坐标计算物体的中心点坐标 (center_x, center_y) 和边界框的宽度和高度 (w, h)，再计算左上角的坐标 (x, y)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    # print(boxes)
                    ell2 = self.geta(boxes)
                    z = z + 1
                    return ell2
        # 非极大值抑制（NMS）
        # 使用 cv2.dnn.NMSBoxes 执行非极大值抑制，去除重叠较多的边界框，保留置信度最高的框。阈值参数分别为置信度阈值（0.5）和 NMS 的 IoU 阈值（0.4）
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in indices:
            box = boxes[i]
            left, top, width, height = box
            right = left + width
            bottom = top + height
            self.draw_prediction(class_ids[i], confidences[i], left, top, right, bottom, frame)

        return None  # 两个眼睛虹膜的中心点坐标

    def get_output_names(self,net):
        # Get the names of all the layers in the network
        layers_names = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        out_layers_indices = net.getUnconnectedOutLayers()
        output_names = [layers_names[i - 1] for i in out_layers_indices]
        return output_names

    def find_rough_border_with_yolov3(self,net, srcw):
        # Convert the image to a blob and set it as input to the network
        # 使用 cv2.dnn.blobFromImage 函数将输入图像 srcw 转换为一个 blob 格式，适用于神经网络
        blob = cv2.dnn.blobFromImage(srcw, 1 / 255.0, (320, 320), [0, 0, 0], True, crop=False)
        net.setInput(blob)

        # Run the forward pass to get output of the output layers
        outs = net.forward(self.get_output_names(net))
        # Remove the bounding boxes with low confidence
        # centers = self.post_process(srcw, outs)
        data1 = self.post_process(srcw,outs)
        return data1
        # print the detected object's center
        # for center in centers:
        #     print(center)
    n = 1

    # -----------------------------------------------
    def EyeProcess(self,config_path,model_path):

        net = self.read_net(config_path, model_path)

        # print(self.img)
        srcw = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        # 用 YOLOv3 模型检测边界
        # print('find_rough_border_with_yolov3 output:')
        data1 = self.find_rough_border_with_yolov3(net, srcw)
        # data = [x_pupil,y_pupil,guanbanx,guanbany,changzhou]
        # print("EyeProcess data:",data)
        return data1

    def geta(self,arr):
        arr_new = []
        arr_new.append(arr[-1])
        for (x1, y1, x2, y2) in arr_new:  # arr为post_process中的boxs,[x,y,w,h](x,y为左上角坐标)
            # print(x1, y1, x2, y2)
            image = self.img
            image = np.array(image)
            # print(f"虹膜 %d 相关参数--------------------------------" %z)
            # 剪裁出虹膜区域
            cropped_image = image[y1:y1 + y2, x1:x1 + x2]
            # 滤波（双边滤波可以很好地平滑图像，同时保留边缘信息，适合处理虹膜等需要保留细节的图像）
            filter_image = cv2.bilateralFilter(cropped_image, 5, 25, 50)
            # cv2.imshow('Original Image', cropped_image)
            # cv2.imshow('Bilateral Filtered Image', filter_image)
            # cv2.waitKey(1111110)

            # 1、瞳孔位置的确定----------------------------------------------------------------------------------------------------
            # 直方图分析求自适应阈值二值化---------------
            hist = cv2.calcHist([filter_image], [0], None, [256], [0, 256])
            # plt.hist(hist)
            # plt.show()
            # 找到像素值在0-30范围内的所有峰值
            peak_range = (0, 40)
            peak_indices = np.where((peak_range[0] <= np.arange(len(hist))) & (np.arange(len(hist)) <= peak_range[1]))[
                0]

            # 找到峰值
            hist_peak = hist[peak_indices]
            max_peak_index = peak_indices[np.argmax(hist_peak)]
            # 确定峰值下降最快的点
            max_descent_index = max_peak_index
            max_descent_rate = -1
            for index in range(max_peak_index, 25):
                descent_rate = hist[index] - hist[index + 1]
                if descent_rate > max_descent_rate:
                    max_descent_rate = descent_rate
                    max_descent_index = index + 1
            if max_descent_index != None:
                global binary_image
                # cv2.threshold 函数用于根据该阈值对滤波后的图像进行二值化,小于阈值的像素设置为 255，其余设为 0
                ret, binary_image = cv2.threshold(filter_image, max_descent_index + 3, 255, type=cv2.THRESH_BINARY_INV)
                # cv2.imshow('Original Image', binary_image)
                # cv2.waitKey(100000)
            # 修复掩膜
            inpaintMask = cv2.threshold(filter_image, 100, 255, cv2.THRESH_BINARY)[1]
            # plt.imshow(inpaintMask,cmap='gray')
            roiTemp = cv2.inpaint(filter_image, inpaintMask, 10, cv2.INPAINT_TELEA)  # 使用INPAINT_TELEA方法
            # plt.imshow(roiTemp,cmap='gray')
            # plt.show()
            mythreshhold = 25  # 从25变到35效果好一些？

            # 直方图均衡化：直方图均衡化(Histogram Equalization) 又称直方图平坦化（增强图像亮度和对比度）
            equalizeHistSrc = cv2.equalizeHist(roiTemp)
            # 阈值分割
            _, equalizeHistSrc = cv2.threshold(equalizeHistSrc, mythreshhold, 255, cv2.THRESH_BINARY_INV)
            # 中值滤波
            equalizeHistSrc = cv2.medianBlur(equalizeHistSrc, 3)
            # plt.imshow(equalizeHistSrc)
            # plt.show()
            # 找轮廓------------------------------
            # contours: 一个 Python 列表，其中包含检测到的轮廓，每个轮廓是一个由点（x, y 坐标）组成的数组
            contours, _ = cv2.findContours(equalizeHistSrc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 有多组contours，找最长的
            global contour
            maxIndex = 0
            maxLen = 0
            k = 0
            # if n == 48:
            #     print(contours)
            for contour in contours:
                if (len(contour)) > maxLen:
                    maxLen = len(contour)
                    maxIndex = k
                k += 1
            if len(contours) == 0:
                continue
            # print(maxIndex)
            # print(len(contours[maxIndex]))
            if len(contours[maxIndex]) > 3:  # 确保轮廓中有足够的点才使用 cv2.minEnclosingCircle 来拟合出包含瞳孔的最小圆，得到瞳孔的中心坐标和半径
                (x, y), radius = cv2.minEnclosingCircle(contours[maxIndex])
                # 计算在原图中的位置
                global x_pupil
                global y_pupil
                x_pupil = x + x1
                y_pupil = y + y1

                # print(f"瞳孔中心坐标（{x_pupil}, {y_pupil}）")
                cv2.circle(image, (int(x_pupil), int(y_pupil)), int(radius), (255, 255, 255), 1)
                # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 转换为RGB用于matplotlib显示
                # plt.title('Image with Circle')
                # plt.show()

                # 均匀选取瞳孔边界的10个点
                pupil_edge_points = []  # 存储瞳孔边界点
                xx = 0
                yy = 0
                for j in range(10):
                    theta = 2 * np.pi * j / 10
                    xx = x_pupil + radius * np.cos(theta)
                    yy = y_pupil + radius * np.sin(theta)
                    pupil_edge_points.append(xx)
                    pupil_edge_points.append(yy)
                    # cv2.circle(image, (int(xx), int(yy)), 3, (0, 255, 0), -1)  # 使用红色点标记
                    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 转换为RGB用于matplotlib显示
                    # plt.title('Image with Circle')
                    # plt.show()
                with open("pupil.txt", "a") as file:
                    # file.write(",".join(map(str, pupil_edge_points))+"\n")  # 转换为字符串并用逗号连接
                    file.write("%d" % xx)
                    file.write(",")
                    file.write("%d" % yy)
                    file.write("\n")
                # with open("pupil.txt", "a") as file:
                #     file.write(",".join(map(str, pupil_edge_points))+"\n")  # 转换为字符串并用逗号连接
                print(f"10个瞳孔边界点：{pupil_edge_points}")

            # 2、光斑位置的确定----------------------------------------------------------------------------------------------------
            # 固定阈值分割

            ret, binary_image1 = cv2.threshold(filter_image, 85, 255, cv2.THRESH_BINARY)
            # cv2.imshow("res_img", binary_image1)  # 局部图的展示
            # cv2.waitKey(100000)
            height, width = binary_image1.shape

            # 找轮廓
            contours1, _ = cv2.findContours(binary_image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 形态学筛选轮廓
            ell = []
            for contour in contours1:
                # 返回一个字典，包含了各种矩特征，例如面积、质心位置、方向等
                moments = cv2.moments(contour)
                area = moments['m00']  # 图像中像素总和

                # print('contour',contour)
                # print(moments)
                if 8 <= len(contour) <= 20:  # 反射光斑的面积大于5个像素小于45个像素
                    # 求轮廓的质心
                    # print("area:", area)
                    x = moments['m10'] / area
                    y = moments['m01'] / area

                    # 利用外接圆圆心求质心
                    # (x, y), radius = cv2.minEnclosingCircle(contour)

                    # 光斑的位置距离虹膜框的边界大于20像素
                    # distance_to_boundary = min(x, y, width - x, height - y)
                    # if distance_to_boundary >= 0:
                    x_spot = x + x1
                    y_spot = y + y1
                    ell.append([x_spot, y_spot])
                        # print(f"光斑中心坐标（{x_spot}, {y_spot}）")
                    # cv2.circle(image, (int(x_spot), int(y_spot)), 0, (0, 0, 255), 2)
                    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 转换为RGB用于matplotlib显示
                    # plt.show()
                if 2 <= len(contour) <= 7:

                    contour = np.array(contour)
                    x = np.mean(contour[:, :, 0])
                    y = np.mean(contour[:, :, 1])
                    # distance_to_boundary = min(x, y, width - x, height - y)
                    # if distance_to_boundary >= 0:
                    x_spot = x + x1
                    y_spot = y + y1
                    ell.append([x_spot, y_spot])
            # print(ell)
            axes = ((x2 // 2 - 5), (y2 // 2 - 5))
            ell2 = []
            for point in ell:
                point = [int(x) for x in point]
                # 判断光斑是否位于拟合的椭圆内部
                if self.is_point_in_ellipse(point, (x1 + x2 // 2, y1 + y2 // 2), axes, 0):
                    ell2.append(point[0])
                    ell2.append(point[1])
                    cv2.circle(image, point, 1, (0, 0, 255), 2)  # 绿色点表示在椭圆内部
                # # else:
                #     cv2.circle(image, point, 5, (0, 0, 255), -1)  # 红色点表示在椭圆外部
            with open("guanban2.txt", "a") as file:
                file.write(",".join(map(str, ell2)) + "\n")  # 转换为字符串并用逗号连接
            # print("各光斑中心:",ell2)
            # 将光斑拟合椭圆,拟合椭圆至少需要5个点
            # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # plt.title('Ellipse')
            # plt.show()
            cv2.imwrite(r"C:\Users\wjy\Documents\WeChat Files\wxid_69cesx39b4dv12\FileStorage\File\2025-05\wSaved\wSaved\photo\frame%d.jpg" %self.n,image)

            # if (len(ell2) >= 5):
            #
            #     ell2 = np.array(ell2)
            #     # print(ell2)
            #     ell2 = ell2.astype(np.int32)
            #     ellips = cv2.fitEllipse(ell2)
            #     global guanbanx
            #     global guanbany
            #     global changzhou
            #     guanbanx = ellips[0][0]
            #     guanbany = ellips[0][1]
            #     changzhou = ellips[1][0]  # 椭圆的长轴长度
                # print('changzhou', changzhou)
                # cv2.ellipse(image, ellips, (0, 255, 0), 2)
                # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                # plt.title('Ellipse')
                # plt.show()
            # else:
            #     guanbanx = None
            #     guanbany = None
            #     changzhou = None
            # # print("guanbanx,guanbany:",guanbanx,guanbany)
            # cen_corneal = [guanbanx,guanbany]
            return ell2
            # import os
            # if not os.path.exists("E:/516/cut3/%d/%s" % (self.m, self.p)):
            #     os.makedirs("E:/516/cut3/%d/%s" % (self.m, self.p))
            # cv2.imwrite("E:/516/cut3/%d/%s/frame%d.jpg" % (self.m, self.p, n),original_img)



    def is_point_in_ellipse(self,point, ellipse_center, axes, angle):
        # 将点转换为椭圆的旋转坐标系
        x, y = point
        cx, cy = ellipse_center
        x_rot = (x - cx) * np.cos(np.radians(-angle)) - (y - cy) * np.sin(np.radians(-angle))
        y_rot = (x - cx) * np.sin(np.radians(-angle)) + (y - cy) * np.cos(np.radians(-angle))

        # 检查点是否满足椭圆方程
        a, b = axes
        return (x_rot ** 2) / (a ** 2) + (y_rot ** 2) / (b ** 2) <= 1

# path = r"D:\视线跟踪\point_matching\eye\image_7.jpg"  # 根据实际情况改
# config_path = "./dense-re.cfg"
# model_path = "./dense-re_24000.weights"
# # #
# imgraw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 0)
# height, width = imgraw.shape
# middle = width // 2
# img1 = imgraw[:, :middle]
# img2 = imgraw[:, middle:]  # 最终在原图上的坐标应该+800像素
# #
# detection1 = Detection(img2)
# #
# ell2,cen_corneal = detection1.EyeProcess(config_path, model_path)
# ell2 = np.array(ell2)
# cen_corneal = np.array([int(cen_corneal[0]),int(cen_corneal[1])])
# print(ell2)
# print(cen_corneal)
# colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
#           (255, 0, 255), (0, 255, 255), (128, 0, 128), (0, 128, 255)]
# img2_cl = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
# for i,(x,y) in enumerate(ell2):
#     cv2.circle(img2_cl,(x,y),radius=5, color=colors[i], thickness=-1)
# cv2.circle(img2_cl,cen_corneal,3,(255,255,255),thickness=-1)
# cv2.imshow('Image with Circles', img2_cl)
# cv2.waitKey(0)  # 等待按键关闭窗口
# cv2.destroyAllWindows()
# print("data1",data1)
#
# detection2 = Detection(img2)
# data2 = detection2.EyeProcess(config_path, model_path)  # 以下加入判断data里面是否有None值（有些帧光斑提取不到）