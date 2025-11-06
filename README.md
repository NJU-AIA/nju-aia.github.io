# 南京大学人工智能协会官网

欢迎访问南京大学人工智能协会（AIA）官方网站。本网站基于 [Astro](https://astro.build/) 框架构建，旨在展示协会动态、活动信息、技术分享等内容，促进校内外人工智能爱好者的交流与学习。


## ✍️ 如何添加文章

1. **进入文章目录**  
   在 `src/content/TechTutorials/` 目录下，添加 Markdown 文件作为技术教程。
   在 `src/content/activityPosts/` 目录下，添加 Markdown 文件作为活动推文。

2. **编写文章内容**  
   每篇文章建议包含如下 Frontmatter（元信息）：

   ```markdown
   ---
   title: 文章标题
   author: 作者名
   description: 文章简要描述
   date: 2025-08-18
   difficulty: "中级"
   readTime: "30 min"
   tags: ["TAG1","TAG2" ]
   ---

   正文内容从这里开始……
   ```

4. **保存并提交**  
   保存文件后，网站会自动识别并展示新文章。可通过 `pnpm run dev` 本地预览效果。
   将修改提交到 github 仓库，网站会自动更新部署。

## 🚀 本地开发与部署

在项目根目录下，使用以下命令：

| 命令                | 作用                         |
|---------------------|------------------------------|
| `pnpm install`       | 安装依赖                     |
| `pnpm run dev`       | 启动本地开发服务器           |
| `pnpm run build`     | 构建生产环境静态文件         |
| `pnpm run preview`   | 预览构建后的站点             |
|` pnpm astro sync` | 生成类型声明（可以消除一些奇怪的报错）|

## 📖 了解更多

- [Astro 官方文档](https://docs.astro.build)
- [南京大学人工智能协会主页](https://nju-aia.github.io)

如有疑问或建议，欢迎提交 Issue 或加入协会交流！
