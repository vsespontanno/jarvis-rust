use crate::command::Command;

pub fn parse(text: &str) -> Command {
    let words = normalized_words(text);

    if asks_for_time(&words) {
        Command::TellTime
    } else {
        Command::Unknown {
            text: text.trim().to_owned(),
        }
    }
}

fn normalized_words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_owned)
        .collect()
}

fn asks_for_time(words: &[String]) -> bool {
    let contains = |expected: &str| words.iter().any(|word| word == expected);
    let contains_which = || ["который", "которые"].iter().any(|word| contains(word));

    (contains("сколько") && (contains("времени") || contains("время")))
        || (contains_which() && contains("час"))
        || ((contains("текущее") || contains("текущий")) && (contains("время") || contains("час")))
        || (contains("скажи") && contains("время"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_supported_time_phrases() {
        for phrase in [
            "сколько времени",
            "Который сейчас час?",
            "которые час",
            "которые сейчас час.",
            "скажи текущее время",
            "Скажи время, пожалуйста!",
        ] {
            assert_eq!(parse(phrase), Command::TellTime, "phrase: {phrase}");
        }
    }

    #[test]
    fn does_not_confuse_duration_with_current_time() {
        assert_eq!(
            parse("поставь таймер на час"),
            Command::Unknown {
                text: "поставь таймер на час".to_owned()
            }
        );
    }

    #[test]
    fn preserves_unknown_text_without_outer_whitespace() {
        assert_eq!(
            parse("  включи музыку  "),
            Command::Unknown {
                text: "включи музыку".to_owned()
            }
        );
    }
}
