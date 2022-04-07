# 一些浅显的笔记
## P0 prepare environment
### Creating a Conda Environment
```shell
conda create --name cs188 python=3.6
```
### Entering the Environment
```shell
conda activate cs188
```
## P1 Search
### Graph Search Pseudo-Code
![image](Image/GraphSearchPseudo-code.png)
### DFS(Depth-First Search)
Time saving(maybe);  &nbsp;Non-optimal solution  
实现思路：按照last-in-first-out(LIFO)的stack容器规则顺序pop出所需的node，然后对node进行判断，如果该node是target node,那么我就记录（返回）到达该node的path  
优点：速度相对较快  
缺点：我最终得到的path不一定是最短路径  
感想：在该课程中，其实基本框架已经搭好，比如说   
- stack容器，
- 判断是否为target node，
- A successor function（这里我的理解是获得我下一步有哪些node可以到达，即两个信息，一是node的位置坐标（state），二是我到达的方式，一般是位移方向，这里指的是东西南北）  
  
  
![image](https://github.com/WhiteFish-gby/CS188_Study/blob/master/Image/dfs.png)

### BFS(Breadth-First Search)

Time Long;   &nbsp;optimal solution  
实现思路：思想与dfs一样，不一样的是采用了queue(FIFO)的数据结构  
优点：可以得到最短路径  
缺点：耗时长
感想：同样queue数据结构等已经被构造好了，我们只要调用就好



![image](https://github.com/WhiteFish-gby/CS188_Study/blob/master/Image/bfs.png)

### UCS(Uniform Cost Search)
optimized form BFS    
UCS是BFS的优化版，即我考虑了到达下一个节点的代价（cost）,我会优先选取cost最小的node  
实现思路：没啥实现思路，cs188已经把PriorityQueue这个class数据结构写好的，我们只需要调用就好了（ps：不是自己写的，感觉还是不太踏实）  
优点：与BFS相比不再是简单的找到target目标，而是将问题更加实际化考虑了行动成本

![image](https://github.com/WhiteFish-gby/CS188_Study/blob/master/Image/ucs.png)

### Greedy Search

![image](https://github.com/WhiteFish-gby/CS188_Study/blob/master/Image/greedy.png)
### A\* Search
![image](https://github.com/WhiteFish-gby/CS188_Study/blob/master/Image/Axing.png)

## P2 Multi-Agent Search
### Reflex Agent
### Minimax
### Alpha-Beta Pruning
### Expectimax Search
## P3 Reinforcement Learning
*生活中很多都是**unexpected***  
*因此我们采用概率表达状态发生的可能性*  
**MDPs(Markov Decision Processes)**  
    &nbsp; MDPs are non-deterministic search problems  
- One way to solve them is with expectimax search  
- We’ll have a new tool soon
  
### Value Iteration
