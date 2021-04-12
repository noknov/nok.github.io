title: Hexo+Nginx搭建静态页面博客
author: meurice
tags: []
categories: []
date: 2020-07-15 15:54:00
---
Hexo 是一个快速、简洁且高效的博客框架，且支持 Markdown 语法。

• 本地环境配置：Node.js+Git+Hexo <br>
• ECS环境配置：(CentOs 8) Node.js+Git+Pm2+Nginx <br>
• 安全组配置：阿里云ECS <br>
• 域名：腾讯云域名解析 <br>
• Webhook：Github

### 本地环境配置 <br>
#### Node.js <br>
　Node.js官网[Node.js](https://nodejs.org/en/)，安装目录尽量不要包括空格，命令行下```node -v```验证是否安装成功。<br>
　或通过[淘宝npm镜像](https://npm.taobao.org/mirrors/node)安装。
#### Git<br>
##### 安装
　Git官网 [Git](https://git-scm.com/)，命令行```git --version```验证。
##### SSH配置
　　配置Github用户名 <br>
  ```
  git config --global user.name "example"
  git config --global user.email "example@email.com"
  ```
　　生成秘钥
  ```
  ssh-keygen -t rsa -b 4096 -C "example@email.com"
  ```

　　~/.ssh文件夹下生成id_rsa.pub公有密钥，依次进入Github——Settings——SSH and GPG keys，添加SSH key，将d_rsa.pub中的内容放入Key中。具体可以参考[这篇文章](https://blog.csdn.net/playboyanta123/article/details/49611873?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase)，Linux系统可以参考[这篇文章](https://blog.csdn.net/qq_36663951/article/details/78749217?utm_medium=distribute.pc_relevant.none-task-blog-baidujs-1)。
#### Hexo<br>
　　npm全局安装
  ```
  sudo npm install -g hexo-cli
  ```

　　验证
 ```
 hexo -version
 ```

　　新建文件夹用来存放Hexo代码，在该文件夹下执行命令行。  
　　初始化Hexo，生成相关文件。
 ```
  hexo -init
 ```
  
　　安装相关依赖
 ```
  npm install
 ```
  
　　预览效果
 ```
  hexo g
  hexo s
 ```
　　浏览器进入http://localhost:4000/
#### 发布到Github<br>
　　安装git部署工具
  ```
  npm install hexo-deployer-git --save
  ```
　　修改_config.yml，repo字段修改为github仓库的SSH链接。
  ```
  deploy:
  type: git
  repo:git@github.com:egname/egrepo.git
  branch: master
  ```
　　代码上传至Github
  ```
  hexo clean
  hexo g
  hexo d
  ```
  ### ECS环境配置 <br>
  
　　服务器为阿里云ECS云服务器，CentOS 8。
   
  #### Node.js<br>
　　采用yum方式安装
   ```
  curl -sL https://rpm.nodesource.com/setup_10.x | bash -
  yum install -y nodejs
  
  node -v # 验证
  ```
  #### Git <br>
  ```
  sudo yum install git
  
  git --version # 验证
  ```
  ##### SSH配置<br>
  ```
  git config --global user.name "example"
  git config --global user.email "example@email.com"
  
  ssh-keygen -t rsa -b 4096 -C "example@email.com"
  ```
　　复制公有密钥，在Github上添加新的SSH key，具体可参考上文本地配置。
  ```
  cd ~/.ssh
  cat id_rsa.pub
  ```
　　从Github仓库中克隆代码。
  ```
  cd /
  mkdir www
 
  cd www
  mkdir blog
  
  cd blog
  git clone git@github.com:egname/egrepo.git
  ```
  #### Nginx <br>
　　安装EPEL存储库
  ```
  sudo yum install epel-release
  ```
  
　　安装Nginx
  ```
  sudo yum install nginx
  ```
　　启动Nginx，设置自启动
  ```
  sudo systemctl start nginx
  sudo systemctl enable nginx
  ```
  ##### Nginx配置
　　进入etc/nginx文件夹下的nginx.conf
  ```
  vi /etc/nginx/nginx.conf
  ```
　　修改配置文件
  ```
  server {
    listen 80;
    server_name www.example.com;
    root /www/blog/example;
    include /etc/nginx/default.d/*.conf;
    
    location / {
      root /www/blog/example;
      index index.jsp index.html index.htm;
    }
  }
  ```
　　重启Nginx
  ```
  systemctl restart nginx
  ```
### 安全组配置
　　以阿里云ECS为例。  
　　进入控制台——网络与安全——安全组，添加入方向规则。  
　　添加端口范围分别为80/80（HTTP），443/443（HTTPS），7777/7777（Webhook），授权对象均为0.0.0.0/0。
  
### 域名解析
　　以腾讯云为例。  
　　DNS 解析 DNSPod——域名解析列表——选择域名——添加记录——快速添加网站解析——指定服务器主机IP（公网）。