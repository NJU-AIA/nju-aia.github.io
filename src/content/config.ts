import { z, defineCollection } from 'astro:content';

export const collections = {
    activityPosts: defineCollection({
        type: "content",
        schema:({image})=> z.object({
            title: z.string(),
            date: z.date(),
            // tags: z.array(z.string()),
            description: z.string().optional(),
            // image: z.string().optional(),
            cover: image(),
        }),
    }),
    TechTutorials: defineCollection({
        type: "content",
        schema: z.object({
            title: z.string(),
            date: z.date(),
            // tags: z.array(z.string()),  // 🚀 必须有 `tags`，否则会报错
            description: z.string().optional(),
            author: z.string(),  // 允许 `author` 存在
            readTime: z.string(),
            difficulty: z.string(),
        }),
    }),
};
