import numpy as np
from skimage import io
from matplotlib import pyplot as plt

import split
import define
import classfiy

import sys
import window
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QFileDialog 

img = 0
img_avg = 0
segments, result, objects = 0, 0, 0
feature = []
n_segments = 0
demoNum = 0
demo = []
trainmat = []
label = []
K = 0
classification = 0
Model = 0
        
def imgPush():
    global mainwindow_ui,img
    imgName, imgType = QFileDialog.getOpenFileName(None,'选择影像文件',".")
    if imgName!="":
        img = io.imread(imgName)
        mainwindow_ui.control_lineEdit.setPlainText("已获取影像，请选择分割方式")

def feature_init():
    global feature
    featureInit = []
    for i in range(n_segments):
        featureInit.append([])
    feature = featureInit

def WatershPush():
    global segments, result, objects, n_segments
    global mainwindow_ui
    mainwindow_ui.control_lineEdit.setPlainText("已选择分水岭分割，请稍等")    
    segments, result, objects = split.WatershedSeg(img)
    n_segments = len(np.unique(segments))
    feature_init()
    io.imshow(result)
    io.show()
    mainwindow_ui.control_lineEdit.appendPlainText("已获取分水岭分割对象\n请关闭分割影像，进行特征提取")

def SlicPush():
    global segments, result, objects, n_segments
    global mainwindow_ui
    mainwindow_ui.control_lineEdit.setPlainText("已选择超像素分割，请稍等")    
    segments, result, objects = split.SlicSeg(img)
    n_segments = len(np.unique(segments))
    feature_init()
    io.imshow(result)
    io.show()
    mainwindow_ui.control_lineEdit.appendPlainText("已获取超像素分割对象\n请关闭分割影像，进行特征提取")

def QuickPush():
    global segments, result, objects, n_segments
    global mainwindow_ui
    mainwindow_ui.control_lineEdit.setPlainText("已选择快速位移分割，请稍等")    
    segments, result, objects = split.QuickSeg(img)
    n_segments = len(np.unique(segments))
    feature_init()
    io.imshow(result)
    io.show()
    mainwindow_ui.control_lineEdit.appendPlainText("已获取快速位移分割对象\n请关闭分割影像，进行特征提取")

def FelzenPush():
    global segments, result, objects, n_segments
    global mainwindow_ui
    mainwindow_ui.control_lineEdit.setPlainText("已选择基于菲尔森茨瓦布高效图分割，请稍等")    
    segments, result, objects = split.FelzenSeg(img)
    n_segments = len(np.unique(segments))
    feature_init()
    io.imshow(result)
    io.show()
    mainwindow_ui.control_lineEdit.appendPlainText("已获取基于菲尔森茨瓦布高效图分割对象\n请关闭分割影像，进行特征提取")

def Feature_avg():    
    global mainwindow_ui,feature
    mainwindow_ui.control_lineEdit.appendPlainText("已选择灰度均值特征\n请选择下一个特征，或者点击确认进入下一步")
    avg = define.gray_mean(objects)
    Avg = define.normalization(avg, n_segments)
    for i in range(n_segments):
        feature[i].extend(Avg[i])

def Feature_sd():
    global mainwindow_ui,feature 
    mainwindow_ui.control_lineEdit.appendPlainText("已选择灰度标准差特征\n请选择下一个特征，或者点击确认进入下一步")
    sd = define.gray_standard_deviation(objects)
    sd = define.normalization(sd, n_segments)
    for i in range(n_segments):
        feature[i].extend(sd[i])

def Feature_max():
    global mainwindow_ui,feature
    mainwindow_ui.control_lineEdit.appendPlainText("已选择灰度最大值特征\n请选择下一个特征，或者点击确认进入下一步")
    max = define.gray_max(objects)
    max = define.normalization(max, n_segments)
    for i in range(n_segments):
        feature[i].extend(max[i])

def Feature_min():
    global mainwindow_ui,feature
    mainwindow_ui.control_lineEdit.appendPlainText("已选择灰度最小值特征\n请选择下一个特征，或者点击确认进入下一步")
    min = define.gray_min(objects)
    min = define.normalization(min, n_segments)
    for i in range(n_segments):
        feature[i].extend(min[i])

def Feature_tex():
    global mainwindow_ui,feature
    mainwindow_ui.control_lineEdit.appendPlainText("已选择纹理特征\n请选择下一个特征，或者点击确认进入下一步")
    tex_histogram = define.texture(img, segments, n_segments)
    tex_histogram = define.normalization(tex_histogram, n_segments)
    for i in range(n_segments):
        feature[i].extend(tex_histogram[i])

def Feature_area():
    global mainwindow_ui,feature
    mainwindow_ui.control_lineEdit.appendPlainText("已选择形状面积特征\n请选择下一个特征，或者点击确认进入下一步")
    areas = define.area(segments)
    areas = define.normalization(areas, n_segments)
    for i in range(n_segments):
        feature[i].extend(areas[i])

def build_avg():
    # 均值图像生成
    avg = define.gray_mean(objects)
    global mainwindow_ui
    mainwindow_ui.control_lineEdit.setPlainText("已生成均值图，请根据此图采样，输入样本")
    mainwindow_ui.control_lineEdit.appendPlainText("demo样本示例(x,y,R,G,B,type):")
    mainwindow_ui.control_lineEdit.appendPlainText("177,282,255,0,0,build\n600,152,255,0,0,build\n830,183,0,255,0,tree\n71,105,0,255,0,tree\n141,407,0,0,255,water\n329,302,0,0,255,water\n783,73,255,0,0,build\n450,355,0,255,0,tree\n91,14,0,0,255,water\n946,206,255,0,0,build\n842,679,0,255,0,tree")  
    global img_avg
    img_avg = img.copy()
    for i in range(len(img_avg)):
        for j in range(len(img_avg[0])):
            k = segments[i][j]
            img_avg[i][j] = avg[k]
    # 显示均值图像
    fig = plt.figure("avg")
    plt.subplot(111)
    plt.title("avg")
    plt.imshow(img_avg)
    plt.show()

def demoIn():
    global mainwindow_ui,demo,trainmat,label,classification,demoNum
    demoIn = mainwindow_ui.demo_line_edit.toPlainText()
    demoInit=[]
    demoInit.extend(demoIn.split("\n"))
    while '' in demoInit:
        demoInit.remove('')
    demoNum=(len(demoInit))

    # 样本赋值到图像
    for i in range(demoNum):
        demo.append(demoInit[i].split(","))
    classification = img_avg.copy()
    id = []  # 样本对象编号
    for i in range(demoNum):
        id.append(segments[int(demo[i][1])][int(demo[i][0])])
        for j in range(len(img_avg)):
            for k in range(len(img_avg[0])):
                if segments[j][k] == id[i]:
                    classification[j][k] = demo[i][2], demo[i][3], demo[i][4]
    # 显示采样结果
    fig = plt.figure("sample")
    plt.subplot(111)
    plt.title("classification")
    plt.imshow(classification)
    mainwindow_ui.control_lineEdit.setPlainText("采样结果,选择分类算法")
    plt.show()
    # 根据样本创建训练集

    for i in range(demoNum):
        trainmat.append(feature[id[i]])  # 样本的特征
        label.append(demo[i][5])  # 样本的属性（树/水）

def Knnchoose():
    global mainwindow_ui
    mainwindow_ui.control_lineEdit.setPlainText("请输入K值（奇数,小于等于样本数）:")

def Knnget_K():
    global mainwindow_ui,K
    K = int(mainwindow_ui.knnK_line_edit.text())
    if K % 2 == 0 or K > demoNum:
        mainwindow_ui.control_lineEdit.appendPlainText("输入有误请重新输入")
        return
    mainwindow_ui.control_lineEdit.setPlainText("已获取K值,请输入模式")
    mainwindow_ui.control_lineEdit.appendPlainText("'uniform'邻域中的所有点的权重相等，'distance'添加距离权重\n")
    mainwindow_ui.control_lineEdit.appendPlainText("点击确认即可进行分类\n亲,分类时间较长，请耐心等待")
def KnnRun():
    global mainwindow_ui
    mainwindow_ui.control_lineEdit.setPlainText("进行Knn分类......")    
    Model = mainwindow_ui.knnModel_line_edit.text()
    type =  classfiy.classification_kNN(n_segments, feature, trainmat, label, K, Model)
    classfiyResult(type)

def SvmRun():
    global mainwindow_ui
    mainwindow_ui.control_lineEdit.setPlainText("进行Svm分类......") 
    type = classfiy.classification_SVM(n_segments, feature, trainmat, label)
    classfiyResult(type)

def classfiyResult(type):
    global feature,trainmat,label
    # 修改像素值，分类完成
    for i in range(len(img_avg)):
        for j in range(len(img_avg[0])):
            pos = np.where(demo == type[segments[i][j]])[0][0]  # 分类结果的样本颜色的位置
            classification[i][j] = demo[pos][2], demo[pos][3], demo[pos][4]
    # 显示最终结果
    fig = plt.figure("classification")
    plt.subplot(111)
    plt.title("classification")
    plt.imshow(classification)
    mainwindow_ui.control_lineEdit.appendPlainText("您已得到分类后的最终结果，谢谢使用，亲\n\n特征已初始化\n若要继续使用请重新进行特征提取！")
    plt.show()
    #初始化
    feature = []
    feature_init()
    trainmat = []
    label = []

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow_ui = window.MainWindow()
    qwidget = QtWidgets.QWidget()
    mainwindow_ui.setupUi(qwidget)
    mainwindow_ui.control_lineEdit.setPlainText("欢迎使用面向对象影像分类小工具\n请打开影像")
    mainwindow_ui.img_push_button.clicked.connect(imgPush)
    mainwindow_ui.division_1.clicked.connect(WatershPush)
    mainwindow_ui.division_2.clicked.connect(SlicPush)
    mainwindow_ui.division_3.clicked.connect(QuickPush)
    mainwindow_ui.division_4.clicked.connect(FelzenPush)
    mainwindow_ui.feature_1.clicked.connect(Feature_avg)
    mainwindow_ui.feature_2.clicked.connect(Feature_sd)
    mainwindow_ui.feature_3.clicked.connect(Feature_max)
    mainwindow_ui.feature_4.clicked.connect(Feature_min)
    mainwindow_ui.feature_5.clicked.connect(Feature_tex)
    mainwindow_ui.feature_6.clicked.connect(Feature_area)
    mainwindow_ui.feature_push.clicked.connect(build_avg)
    mainwindow_ui.demo_push_button.clicked.connect(demoIn)   
    mainwindow_ui.knnAlgorithm.clicked.connect(Knnchoose)  
    mainwindow_ui.knnK_push_button.clicked.connect(Knnget_K)  
    mainwindow_ui.knnModel_push_button.clicked.connect(KnnRun)  
    mainwindow_ui.svmAlgorithm.clicked.connect(SvmRun)  
    qwidget.show()
    sys.exit(app.exec_())