// @ts-check
import { defineConfig } from 'astro/config';

const repository = process.env.GITHUB_REPOSITORY?.split('/')[1];
const owner = process.env.GITHUB_REPOSITORY_OWNER;
const isPagesBuild = process.env.ASTRO_GITHUB_PAGES === 'true';
const configuredSite = process.env.ASTRO_SITE;
const configuredBase = process.env.ASTRO_BASE;
const isProjectPage = repository && !repository.endsWith('.github.io');
const pagesSite = repository?.endsWith('.github.io')
	? `https://${repository}`
	: owner
		? `https://${owner}.github.io`
		: undefined;

// https://astro.build/config
export default defineConfig({
	site: configuredSite ?? (isPagesBuild ? pagesSite : undefined),
	base: configuredBase ?? (isPagesBuild && isProjectPage ? `/${repository}` : undefined),
});
