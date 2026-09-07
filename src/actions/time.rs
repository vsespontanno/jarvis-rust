use jiff::Zoned;

pub fn current_time_message() -> String {
    let now = Zoned::now();
    format_time(now.hour(), now.minute())
}

fn format_time(hour: i8, minute: i8) -> String {
    format!("Сейчас {hour:02}:{minute:02}.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_time_with_leading_zeroes() {
        assert_eq!(format_time(9, 5), "Сейчас 09:05.");
    }
}
