title: Linux设置定时任务
author: meurice
date: 2020-07-16 19:40:44
tags:
---
环境：CentOS 8
### 执行内容
　　新建文件_crond.sh，作为定时执行的内容。
  ```
  #!/bin/bash
  cd /www/blog/hexo
  git pull git@github.com:egname/egrepo.git

  #echo pull successfully > /home/gitpull.log
  ```
  
### crontab服务
　　启动crontab服务，CentOS版本不同，具体命令可能有所差异。
  ```
  systemctl start crond 
  ```
　　启动服务
  
  ```
  systemctl stop crond            # 关闭服务
  systemctl restart crond         # 重启服务     
  systemctl reload crond          # 重新载入配置
  systemctl status crond          # 状态
  ```
### 设置计时器
　　crontab 选项 参数  
　　选项:  
　　　　-e：编辑该用户的计时器设置；  
　　　　-l：列出该用户的计时器设置；  
　　　　-r：删除该用户的计时器设置；  
　　　　-u：指定要设定计时器的用户名称。
  ```
  crontab -e
  ```
  
 　　进入insert插入模式，以每五分钟执行一次为例。ESC后输入wq保存并退出。
  ```
  */5 * * * * /root/_crond.sh
  ```
  <br>
  
  关于Crontab更多具体用法，您可以参考[这篇文章](https://www.cnblogs.com/muscles/p/9532451.html)。