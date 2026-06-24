import { defineCollection, z } from 'astro:content';

const posts = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    date: z.date(),
    description: z.string().optional(),
    tags: z.array(z.string()).optional(),
    status: z.enum(['draft', 'public']).default('public'),
    artifacts: z.array(z.object({
      label: z.string(),
      path: z.string(),
    })).optional(),
    metrics: z.array(z.object({
      label: z.string(),
      value: z.union([z.string(), z.number()]),
    })).optional(),
  }),
});

export const collections = { posts };
