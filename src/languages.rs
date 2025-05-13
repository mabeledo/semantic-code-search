use code_splitter::{Splitter, WordCounter};
use std::sync::OnceLock;

static LANGUAGES: OnceLock<Vec<Language>> = OnceLock::new();

#[allow(dead_code)]
pub struct Language {
    pub name: String,
    pub extensions: Vec<String>,
    pub splitter: Splitter<WordCounter>,
}

fn init_languages() -> Vec<Language> {
    let javascript: Language = Language {
        name: "javascript".to_string(),
        extensions: vec!["js".to_string()],
        splitter: Splitter::new(
            tree_sitter::Language::new(tree_sitter_javascript::LANGUAGE),
            WordCounter,
        )
        .unwrap(),
    };
    let rust: Language = Language {
        name: "rust".to_string(),
        extensions: vec!["rs".to_string()],
        splitter: Splitter::new(
            tree_sitter::Language::new(tree_sitter_rust::LANGUAGE),
            WordCounter,
        )
        .unwrap(),
    };
    let python: Language = Language {
        name: "python".to_string(),
        extensions: vec!["py".to_string()],
        splitter: Splitter::new(
            tree_sitter::Language::new(tree_sitter_python::LANGUAGE),
            WordCounter,
        )
        .unwrap(),
    };
    let typescript: Language = Language {
        name: "typescript".to_string(),
        extensions: vec!["ts".to_string()],
        splitter: Splitter::new(
            tree_sitter::Language::new(tree_sitter_typescript::LANGUAGE_TYPESCRIPT),
            WordCounter,
        )
        .unwrap(),
    };

    vec![javascript, rust, python, typescript]
}

pub fn get_languages() -> &'static [Language] {
    LANGUAGES.get_or_init(init_languages).as_slice()
}
