import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';

export default defineConfig({
  site: 'https://aeshma-daeva.github.io',
  base: '/Demian-Lab',
  integrations: [mdx()],
});
