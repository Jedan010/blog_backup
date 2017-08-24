---
title: Git基础操作
tags:
	- git
	- github
categories:
	- git

---

# Git基础操作

## 介绍
git是什么呢？git是一个分布式版本控制系统，以便于追踪改变和多人协作。

为什么我们要关注版本控制呢？当你写完一篇文章后，如果我们想要修改，又不想破坏原文件，以便修改失误后能还原，我们一般会复制一份原文件，然后再修改。可是如果我们有很多份文件要修改呢？如果我们要修改很多次呢？那每次修改都复制一份出来就显得很繁琐。通过版本控制，我们就可以在一个文件中修改，并能很好的在各个修改的版本中来回转换。

为什么是分布式呢？分布式是相对于集中式的。集中式就是把文件集中放在一个地方，如果需要修改，就把要修改的文件取出来，修改完后将修改的部分再替换回去。这个有两点不好，一是如果集中放置文件的主机出问题了，那所有文件都找不回来了。二是修改必须是联网，如果主机网络出问题，那其他人也不能修改。而分布式则是将文件放在所有需要的人主机上，这样就相当于将文件备份了无数份，几乎没有丢失的可能。另外，所有人都可以在自己的本级上修改，只需要在修改完之后联网传送修改好的版本即可。

## 基础命令

### 获取git仓库

- 从现有目录中初始化仓库
- 从服务器克隆现有仓库

#### 从现有目录初始化仓库
只需要进入需要初始化的仓库，然后输入：
```
$ git init
```
该命令将创建一个名为 .git 的子目录，git所有重要文件都放在这个文件夹里面。

#### 从服务器克隆现有仓库
克隆的命令是`git clone [url]`。比如，要克隆 Git 的可链接库 libgit2，可以用下面的命令：
```
$ git clone https://github.com/libgit2/libgit2
```
这样就把现有的仓库复制到自己的主机上，并且这个文件夹会自动生成.git文件夹。


### 记录每次更新到仓库
在工作目录下的文件只有两种状态：已追踪和未追踪。未追踪的是指没有纳入git版本控制的文件，而已追踪的则是纳入了版本控制的文件，可以用git来进行版本的控制。而纳入了版本控制的已追踪文件有三种状态：未修改、已修改、已放入暂存区。初次克隆某个仓库的时候，工作目录中的所有文件都是已追踪的，并处于未修改状态。

编辑过某些文件后，git会将它们标记会已修改文件。等我们修改好之后，将文件放入暂存区，则git会将这些文件标记为已暂存。最后，我们将暂存的文件提交。这样我们就将记录更新到仓库了，并且此时，git会将这些文件标记为未修改。

![文件的状态变化周期](https://git-scm.com/book/en/v2/images/lifecycle.png)

所以，我们修改完文件后，先提交到暂存区，然后再从暂存区提交到我们的仓库。为什么中间要多加一个暂存区，不直接将修改好的文件添加到仓库呢？要理解暂存区的作用，我们先看看暂存区是怎么工作的。当我们把修改的文件提交到暂存区后，该文件就被标记为已暂存。可是这时你还没修改完，又修改了一部分。那此时这个文件会被标记为已暂存和未修改。已暂存指的是之前暂存的文件，未修改指的是后来又修改后的文件。这样我们就同时记录了两次修改的文件，以便于我们选择我们需要文件。当我们再把已修改的文件提交到暂存区，此时这次文件就只有一个状态——已暂存，而且已暂存的时候最后一次提交的文件。当我们把所有修改好的文件都提交到暂存区后，就可以将暂存区的文件一次全部提交到仓库。所以，暂存区有两个好处，一是进行快照，便于退回；二是分批、分阶段提交。

#### git命令
- 加入暂存区：`git add`
- 提交仓库：`git commit`
- 查询状态：`git status`
- 查询修改：`git diff`
- 移除文件：`git rm`

##### 加入暂存区：
`git add [files]`命令有两个作用，一是将未追踪的文件纳入已追踪中，二是将已修改的文件提交到暂存区。

##### 提交仓库：
`git commit -m "your description"`命令是将暂存区的所有文件都提交到仓库中，并添加一段描述以便于很好的识别这个版本。


##### 查询状态
`git status`命令是用于检查当前状态。

未修改时显示为：
```
$ git status
On branch master
nothing to commit, working directort clean
```
现在创建一个新的`README.md`文件，则该文件属于未追踪状态，则输出为：
```
$ git status
On branch master
Untracked files:
  (use "git add <file>..." to include in what will be committed)

    README

nothing added to commit but untracked files present (use "git add" to track)
```
将文件纳入追踪系统，并放入暂存状态后，输入为：
```
$ git add README
$ git status
On branch master
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

    new file:   README.md
```
修改`README.md`文件后，输出为：
```
$ git status
On branch master
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

    modified:   README.md
```

##### 查询修改

`git diff`用于查看工作目录中当前文件和暂存区域快照之间的差异，也就是修改之后还没有暂存起来的变化内容。

`git diff --cached`或`git diff --staged`可以用来查看已暂存的文件与仓库中文件差异。

##### 移除文件
`git rm <file>`可以将某个文件从仓库中移除，即从已追踪状态转换为未追踪状态，并且从工作目录中将该文件删除，这样以后该文件就不会再出现在未追踪清单中。如果，该文件已放入暂存区，则需要使用`git rm <file> -f`命令来强制移除该文件。

### 撤销操作

 - 重新提交：`git commit --amend`
 - 取消暂存文件：`git reset HEAD <file>`
 - 撤销对文件的修改：`git checkout -- <file>`

#### 重新提交
当我们提交文件时发现漏了几个文件没有添加，或者提交信息写错了，此时用`git commit --amend`可以尝试重新提交，则这次的提交会替代上次的提交。

#### 取消暂存文件
`git reset HEAD <file>`会将文件从暂存区取出来，即该文件的状态从已暂存改为已修改。

#### 撤销对文件的修改
`git checkout -- <file>`撤销对文件的修改，即将修改过文件退回到上一次提交时的状态，即从已修改状态转为未修改状态。


### 远程仓库的使用

- 添加远程仓库：`git remote add`
- 从远程仓库中抓取的拉取：`git fetch` or `git pull`
- 推送到远程仓库：`git push`
- 查看远程仓库：`git remote show`
- 远程仓库的移除与重命名：`git rm` or `git rename`

#### 添加远程仓库
`git remote add <shortname> <url>`可以添加一个新的远程仓库，同时指定一个可以轻松引用的简写。

#### 从远程仓库中抓取的拉取
`git fetch [remote-name]`可以从远程仓库中获取数据，但它并不会自动合并或修改你当前的工作。当准备好的时候，必须手动将其合并入你的工作。

`git pull`可以实现自动抓取并合并远程分支到当前分支。

####  推送到远程仓库
`git push [remote-name] [brach-name]`可以将本地分支推送到远程仓库的某个服务器中，这个只有当你对该服务器具有写入权限，并之前没有人推送过才有用。

当之前有人推送过，你的推送就会被拒绝。你必须先用`git pull`将他们的工作拉取下拉合并到你的分支中，才能推送。

#### 查看远程仓库
`git remote <url>`可以查看已经配置的远程仓库的服务器，如果你克隆了自己的仓库，那么至少能看到`origin`，这是git给你仓库服务器的默认名字。

`git remote -v`则会显示需要读写远程仓库所需要使用的git保存的简写即对应的URL。

`git remore show [remote-name]`除了会列出远程仓库的URL，还列出跟踪分支的信息。告诉你在特定的分支上执行`git push`会自动推送到哪一个远程分支，哪些远程分支不在你的本地，哪些远程分支已经从服务器移除，当你执行`git pull`时，哪些分支会自动合并。

#### 远程仓库的移除与重命名

`git remote rm <remote-name>`可以移除一个远程仓库，`git remote rename <remote-name> <new-name>`可以将远程仓库重命名。



参考:https://git-scm.com/book/zh/v2/
