use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Glossary {
    terms: Vec<&'static str>,
}

impl Default for Glossary {
    fn default() -> Self {
        Self {
            terms: vec![
                "HO", "BZ", "RZ", "Ava", "IP", "spec", "fame", "gank", "rat", "dive", "clap",
                "blob", "wipe",
            ],
        }
    }
}

impl Glossary {
    pub fn protect(&self, text: &str) -> ProtectedText {
        let mut protected = text.to_string();
        let mut replacements = Vec::new();

        for term in &self.terms {
            let token = format!("__ALBION_TERM_{}__", replacements.len());
            let next = replace_ascii_word(&protected, term, &token);
            if next != protected {
                replacements.push((token, (*term).to_string()));
                protected = next;
            }
        }

        ProtectedText {
            text: protected,
            replacements,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtectedText {
    pub text: String,
    replacements: Vec<(String, String)>,
}

impl ProtectedText {
    pub fn restore(&self, translated: &str) -> String {
        let mut restored = translated.to_string();
        for (token, term) in &self.replacements {
            restored = restored.replace(token, term);
        }

        restored
    }

    pub fn replacement_map(&self) -> HashMap<&str, &str> {
        self.replacements
            .iter()
            .map(|(token, term)| (token.as_str(), term.as_str()))
            .collect()
    }
}

fn replace_ascii_word(text: &str, needle: &str, replacement: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut last = 0;

    for (index, _) in text.match_indices(needle) {
        let before = text[..index].chars().next_back();
        let after = text[index + needle.len()..].chars().next();

        if before.is_none_or(|c| !is_word_char(c)) && after.is_none_or(|c| !is_word_char(c)) {
            output.push_str(&text[last..index]);
            output.push_str(replacement);
            last = index + needle.len();
        }
    }

    output.push_str(&text[last..]);
    output
}

fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_game_terms_as_whole_words() {
        let glossary = Glossary::default();
        let protected = glossary.protect("hola HO gank fame spec");
        let translated = protected.restore(&protected.text.replace("hola", "hello"));

        assert_eq!(translated, "hello HO gank fame spec");
    }

    #[test]
    fn does_not_replace_inside_words() {
        let glossary = Glossary::default();
        let protected = glossary.protect("famed ganker");

        assert_eq!(protected.text, "famed ganker");
    }
}
