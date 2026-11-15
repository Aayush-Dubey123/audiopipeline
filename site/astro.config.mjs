// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	// Set `site` (and `base` for a GitHub Pages project site) when deploying, e.g.
	// site: 'https://aayush-dubey123.github.io', base: '/audiopipeline'
	integrations: [
		starlight({
			title: 'AudioPipeline',
			description:
				'Hindi & English voice cloning on a 6 GB laptop GPU: what I built, what I measured, and what I learned.',
			logo: { src: './src/assets/logo.svg', replacesTitle: false },
			favicon: '/favicon.svg',
			customCss: ['@fontsource-variable/inter', './src/styles/theme.css'],
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/Aayush-Dubey123/audiopipeline' }],
			lastUpdated: false,
			tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 3 },
			sidebar: [
				{
					label: 'The story',
					items: [
						{ label: 'The problem', slug: 'story/problem' },
						{ label: 'How it works', slug: 'story/how-it-works' },
						{ label: 'What I fixed', slug: 'story/fixes' },
						{ label: 'Listen: before & after', slug: 'story/listen' },
					],
				},
				{
					label: 'Engineering',
					items: [
						{ label: 'Built on my first GPU', slug: 'engineering/first-gpu', badge: { text: 'RTX 3050', variant: 'success' } },
						{ label: 'Tuning by measurement', slug: 'engineering/tuning' },
						{ label: 'Why it isn’t online 24/7', slug: 'engineering/hosting' },
					],
				},
				{
					label: 'Use it',
					items: [
						{ label: 'Run it yourself', slug: 'guides/run-it-yourself' },
						{ label: 'Responsible use', slug: 'guides/responsible-use' },
					],
				},
				{
					label: 'Learn',
					items: [
						{ label: 'Audio basics', slug: 'learn/audio-basics' },
						{ label: 'Lessons & what’s next', slug: 'learn/lessons' },
					],
				},
			],
		}),
	],
});
