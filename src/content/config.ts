import { z, defineCollection } from 'astro:content';

export const collections = {
    posts: defineCollection({
        type: "content",
        schema: z.object({
            title: z.string(),
            tags: z.array(z.string()),
            date: z.date(),
        }),
    }),
    // images: defineCollection({
    //     type: "data", // 由于 images 不是 markdown 内容，使用 "data"
    //     schema: null, // 可以不需要 schema
    // }),
};
 
//  This code defines a collection of blog posts. Each blog post has a title and an array of tags. 
//  Now, you can use this collection in your pages.