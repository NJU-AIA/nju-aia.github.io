import { z, defineCollection } from 'astro:content';

export const collections = {
  activityPosts: defineCollection({
    type: 'content',
    schema: ({ image }) =>
      z.object({
        title: z.string(),
        date: z.date(),
        description: z.string().optional(),
        cover: image(), // 你的原逻辑保留
      }),
  }),

  TechTutorials: defineCollection({
    type: 'content',
    schema: z.object({
      title: z.string(),
      date: z.date(),
      description: z.string().optional(),
      author: z.string(),
      readTime: z.string(),
      difficulty: z.string(),
    }),
  }),

  // ✅ 新增的 slides 集合（放 Slidev 的 .md）
  slides: defineCollection({
    type: 'content',
    schema: z.object({
      title: z.string(),
      date: z.date(),               // frontmatter 用 ISO 或 yyyy-mm-dd，Astro 会解析成 Date
      description: z.string().optional(),
      slug: z.string().optional(),  // 不填则用文件名
      cover: z.string().optional(), // 建议用绝对路径，比如 /images/xxx.png，方便 Slidev 直接引用
    }),
  }),
};
