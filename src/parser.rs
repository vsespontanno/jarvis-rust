use crate::command::Command;

pub const VERSION: u32 = 1;

pub fn parse(text: &str) -> Command {
    let words = normalized_words(text);

    if is_non_speech_annotation(text, &words) {
        Command::NoSpeech
    } else if asks_for_time(&words) {
        Command::TellTime
    } else if let Some(duration_seconds) = timer_duration_seconds(&words) {
        Command::SetTimer { duration_seconds }
    } else if asks_to_play_music(&words) {
        Command::PlayMusic
    } else {
        Command::Unknown {
            text: text.trim().to_owned(),
        }
    }
}

const MAX_TIMER_SECONDS: u64 = 24 * 60 * 60;

fn timer_duration_seconds(words: &[String]) -> Option<u64> {
    let contains_any = |expected: &[&str]| {
        words
            .iter()
            .any(|word| expected.iter().any(|expected| word == expected))
    };
    let set_actions = ["поставь", "поставьте", "поставить"];
    let has_action = contains_any(&[
        "поставь",
        "поставьте",
        "поставить",
        "запусти",
        "запустить",
        "включи",
        "включить",
    ]);
    let has_timer = contains_any(&["таймер", "таймера", "тайга"]);
    let has_implicit_timer = words
        .windows(2)
        .any(|pair| set_actions.contains(&pair[0].as_str()) && pair[1] == "на");
    if !has_action || (!has_timer && !has_implicit_timer) {
        return None;
    }

    let mut total_seconds = 0_u64;
    for (index, word) in words.iter().enumerate() {
        let multiplier = match word.as_str() {
            "секунда" | "секунду" | "секунды" | "секунд" => 1,
            "минута" | "минуту" | "минуты" | "минут" => 60,
            "час" | "часа" | "часов" => 60 * 60,
            _ => continue,
        };
        let amount = number_before(words, index)?;
        total_seconds = total_seconds.checked_add(amount.checked_mul(multiplier)?)?;
    }

    (1..=MAX_TIMER_SECONDS)
        .contains(&total_seconds)
        .then_some(total_seconds)
}

fn number_before(words: &[String], end: usize) -> Option<u64> {
    let last = words.get(end.checked_sub(1)?)?.as_str();
    if last == "на" {
        return Some(1);
    }
    if let Ok(number) = last.parse() {
        return Some(number);
    }

    let last_value = number_word(last)?;
    if last_value < 10
        && let Some(tens) = end
            .checked_sub(2)
            .and_then(|index| words.get(index))
            .and_then(|word| tens_word(word))
    {
        return Some(tens + last_value);
    }
    Some(last_value)
}

fn number_word(word: &str) -> Option<u64> {
    match word {
        "один" | "одна" | "одну" => Some(1),
        "два" | "две" => Some(2),
        "три" => Some(3),
        "четыре" => Some(4),
        "пять" => Some(5),
        "шесть" => Some(6),
        "семь" => Some(7),
        "восемь" => Some(8),
        "девять" => Some(9),
        "десять" => Some(10),
        "одиннадцать" => Some(11),
        "двенадцать" => Some(12),
        "тринадцать" => Some(13),
        "четырнадцать" => Some(14),
        "пятнадцать" => Some(15),
        "шестнадцать" => Some(16),
        "семнадцать" => Some(17),
        "восемнадцать" => Some(18),
        "девятнадцать" => Some(19),
        _ => tens_word(word),
    }
}

fn tens_word(word: &str) -> Option<u64> {
    match word {
        "двадцать" => Some(20),
        "тридцать" => Some(30),
        "сорок" => Some(40),
        "пятьдесят" => Some(50),
        "шестьдесят" => Some(60),
        "семьдесят" => Some(70),
        "восемьдесят" => Some(80),
        "девяносто" => Some(90),
        _ => None,
    }
}

fn is_non_speech_annotation(text: &str, words: &[String]) -> bool {
    let trimmed = text.trim();
    let fully_wrapped = [('[', ']'), ('(', ')'), ('*', '*')]
        .iter()
        .any(|&(start, end)| trimmed.starts_with(start) && trimmed.ends_with(end));

    (!words.is_empty() && fully_wrapped)
        || (words.len() == 1 && matches!(words[0].as_str(), "музыка" | "шум" | "тишина"))
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
        "открою",
        "открыть",
    ]);
    let has_music = contains_any(&["музыка", "музыку", "spotify", "спотифай", "спотик"]);

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
    fn recognizes_timer_for_one_hour_without_an_explicit_number() {
        assert_eq!(
            parse("поставь таймер на час"),
            Command::SetTimer {
                duration_seconds: 3_600
            }
        );
    }

    #[test]
    fn recognizes_timer_durations_and_units() {
        for (phrase, duration_seconds) in [
            ("поставь таймер на 30 секунд", 30),
            ("поставьте на одну минуту", 60),
            ("поставь тайга на 1 минуту", 60),
            ("поставь таймер на пять минут", 300),
            ("запусти таймер на двадцать пять минут", 1_500),
            ("включи таймер на 1 час 30 минут", 5_400),
        ] {
            assert_eq!(
                parse(phrase),
                Command::SetTimer { duration_seconds },
                "phrase: {phrase}"
            );
        }
    }

    #[test]
    fn does_not_create_timer_without_duration_or_action() {
        for phrase in [
            "поставь таймер",
            "таймер на пять минут",
            "поставь музыку на одну минуту",
        ] {
            assert!(matches!(parse(phrase), Command::Unknown { .. }));
        }
    }

    #[test]
    fn recognizes_supported_music_phrases() {
        for phrase in [
            "включи музыку",
            "Запусти музыку, пожалуйста",
            "открой Spotify",
            "Открою Spotify",
            "включить спотифай",
            "включи спотик",
        ] {
            assert_eq!(parse(phrase), Command::PlayMusic, "phrase: {phrase}");
        }
    }

    #[test]
    fn recognizes_whisper_non_speech_annotations() {
        for transcript in [
            "[музыка]",
            "[ШУМ]",
            "тишина.",
            "[звук от моего слоя]",
            "(звук от джанра)",
            "*хм*",
            "*клап* *клап*",
        ] {
            assert_eq!(parse(transcript), Command::NoSpeech);
        }
    }

    #[test]
    fn wrapped_annotation_never_executes_a_command() {
        assert_eq!(parse("[включи музыку]"), Command::NoSpeech);
        assert_eq!(parse("(который час)"), Command::NoSpeech);
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
