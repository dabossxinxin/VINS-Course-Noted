**作者**：
贺一家，高翔，崔华坤，赵松

**描述**：
这个代码用来演示VIO算法中的非线性优化过程、滑动窗过程以及测试VIOSLAM中的一系列实用trip

**依赖项**
1. pangolin: <https://github.com/stevenlovegrove/Pangolin>

2. opencv

3. Eigen

4. Ceres: vins初始化部分使用了ceres做sfm，所以我们还是需要依赖ceres. 

### 编译代码

代码测试环境为windows10 & VS2015

### 运行
#### 1. CurveFitting Example to Verify Our Solver.
```c++
cd build
../bin/testCurveFitting 
```

#### 2. VINs-Mono on Euroc Dataset
```c++
cd build
../bin/run_euroc /home/dataset/EuRoC/MH-05/mav0/ ../config/
```

#### 3. VINs-Mono on Simulation Dataset (project homework)

you can use this code to generate vio data.

https://github.com/HeYijia/vio_data_simulation