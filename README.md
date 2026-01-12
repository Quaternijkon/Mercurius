<!--
 * @Author: quaternijkon quaternijkon@mail.ustc.edu.cn
 * @Date: 2025-12-18 08:45:06
 * @LastEditors: quaternijkon quaternijkon@mail.ustc.edu.cn
 * @LastEditTime: 2026-01-12 08:11:11
 * @FilePath: /faiss/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
#

当前工作分支为`single`

最新进展代码为`demo7-hierarchy.cpp`

使用的数据集位于老GPU节点(222.195.68.87:11451)`/home/gpu/dry/Mercurius/faiss/sift`
或者通过以下方式获取：

```shell
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz && tar -zxvf sift.tar.gz && rm sift.tar.gz
```

运行方式：在根目录(README.md所在目录)运行`make run`

--- 

适用于大数据集的相同实现以及使用的数据集在A40-node1上：`/home/taig/dry/faiss/sift`