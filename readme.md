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
Time saving(maybe);  
stick to one's ways(提条路走到黑);  
Non-optimal solution  
![image](https://github.com/WhiteFish-gby/CS188_Study/blob/master/Image/dfs.png)

### BFS(Breadth-First Search)

Time Long;  
optimal solution  
![image](https://github.com/WhiteFish-gby/CS188_Study/blob/master/Image/bfs.png)

### UCS(Uniform Cost Search)
optimized form BFS  
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
