# 编译器设置
CXX = g++

# 编译选项: C++标准, 优化等级, OpenMP支持, 头文件路径
CXXFLAGS = -std=c++17 -O3 -fopenmp -I ./

# 链接选项: 库路径, RPATH (运行时库路径)
LDFLAGS = -L ./build/faiss -Wl,-rpath,./build/faiss

# 需要链接的库
LDLIBS = -lfaiss -lopenblas

# 定义目标名称
TARGET1 = demo
TARGET2 = demo2-allIVF
TARGET3 = demo3-nprobe
TARGET4 = demo4-hnsw
TARGET5 = demo4-hnsw-exclude
TARGET6 = demo4-hnsw-reference
TARGET7 = demo5-flat
TARGET8 = demo6-latency
TARGET9 = demo7-hierarchy

# 默认目标: 同时编译所有可执行文件
all: $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5) $(TARGET6) $(TARGET7) $(TARGET8) $(TARGET9)

# ==========================================
# 通用编译规则
# ==========================================
# 任何不带后缀的目标 (如 demo3-nprobe)，都由对应的 .cpp 文件编译而来
%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# ==========================================
# 运行规则
# ==========================================

# 运行 demo
run-demo: $(TARGET1)
	./$(TARGET1)

# 运行 demo2
run-demo2: $(TARGET2)
	./$(TARGET2)

# 运行 demo3 (新增)
run-demo3: $(TARGET3)
	./$(TARGET3)

run-demo4: $(TARGET4)
	./$(TARGET4)

run-demo5: $(TARGET5)
	./$(TARGET5)

run-demo6: $(TARGET6)
	./$(TARGET6)

run-demo7: $(TARGET7)
	./$(TARGET7)

run-demo8: $(TARGET8)
	./$(TARGET8)

run-demo9: $(TARGET9)
	./$(TARGET9)

# 默认运行最新的 demo3
run: run-demo9

# ==========================================
# 清理规则
# ==========================================
clean:
	rm -f $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(TARGET5) $(TARGET6) $(TARGET7) $(TARGET8) $(TARGET9)

.PHONY: all run run-demo run-demo2 run-demo3 run-demo4 run-demo5 run-demo6 run-demo7 clean