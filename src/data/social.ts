export interface SocialLink {
	friendlyName: string;
	link: string;
	/** Iconify name, e.g. "simple-icons:github". */
	icon: string;
}

export const socialLinks: SocialLink[] = [
	{
		friendlyName: "Email",
		link: "mailto:spacetime0311@gmail.com",
		icon: "mdi:email-outline",
	},
	{
		friendlyName: "Threads",
		link: "https://www.threads.net/@xavier.data.forge",
		icon: "simple-icons:threads",
	},
	{
		friendlyName: "Github",
		link: "https://github.com/xavierforge",
		icon: "simple-icons:github",
	},
	{
		friendlyName: "LinkedIn",
		link: "https://www.linkedin.com/in/chih-ying-yen/",
		icon: "simple-icons:linkedin",
	},
];
