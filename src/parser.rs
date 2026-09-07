use crate::command::Command;

pub fn parse(text: &str) -> Command {
    let words = normalized_words(text);

    if is_non_speech_annotation(&words) {
        Command::NoSpeech
    } else if asks_for_time(&words) {
        Command::TellTime
    } else if asks_to_play_music(&words) {
        Command::PlayMusic
    } else {
        Command::Unknown {
            text: text.trim().to_owned(),
        }
    }
}

fn is_non_speech_annotation(words: &[String]) -> bool {
    words.len() == 1 && matches!(words[0].as_str(), "музыка" | "шум" | "тишина")
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

fn asks_to_play_music(words: &[String]) -> bool {
    let contains_any = |expected: &[&str]| {
        words
            .iter()
            .any(|word| expected.iter().any(|expected| word == expected))
    };

    let has_action = contains_any(&[
        "включи",
        "включить",
        "запусти",
        "запустить",
        "открой",
        "открыть",
    ]);
    let has_music = contains_any(&["музыка", "музыку", "spotify", "спотифай"]);

    has_action && has_music
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
    fn recognizes_supported_music_phrases() {
        for phrase in [
            "включи музыку",
            "Запусти музыку, пожалуйста",
            "открой Spotify",
            "включить спотифай",
        ] {
            assert_eq!(parse(phrase), Command::PlayMusic, "phrase: {phrase}");
        }
    }

    #[test]
    fn recognizes_whisper_non_speech_annotations() {
        for transcript in ["[музыка]", "[ШУМ]", "тишина."] {
            assert_eq!(parse(transcript), Command::NoSpeech);
        }
    }

    #[test]
    fn does_not_hide_annotations_inside_real_phrases() {
        assert_eq!(
            parse("слышен шум вентилятора"),
            Command::Unknown {
                text: "слышен шум вентилятора".to_owned()
            }
        );
    }

    #[test]
    fn mentioning_music_without_an_action_is_not_a_command() {
        assert_eq!(
            parse("музыка помогает работать"),
            Command::Unknown {
                text: "музыка помогает работать".to_owned()
            }
        );
    }

    #[test]
    fn preserves_unknown_text_without_outer_whitespace() {
        assert_eq!(
            parse("  расскажи анекдот  "),
            Command::Unknown {
                text: "расскажи анекдот".to_owned()
            }
        );
    }
}
