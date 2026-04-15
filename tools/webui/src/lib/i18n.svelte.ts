import { translations, type Language, type TranslationKeys } from './i18n';
import { app } from './state.svelte';

/**
 * Get translated string by key
 * @param key - Translation key
 * @returns Translated string or key if not found
 */
export function t(key: TranslationKeys): string {
	const lang: Language = app.lang as Language;
	return translations[lang]?.[key] || translations.en[key] || key;
}

/**
 * Get current language
 */
export function getLang(): Language {
	return app.lang as Language;
}

/**
 * Set language and trigger reactivity
 */
export function setLang(lang: Language): void {
	app.lang = lang;
}

/**
 * Toggle between English and Chinese
 */
export function toggleLang(): void {
	app.lang = app.lang === 'en' ? 'zh' : 'en';
}
