//! Minimal multipart/form-data body splitter, matching the hand-rolled
//! boundary-splitting logic server.py used (search for the first part whose
//! headers mention `filename`, take everything after the blank line up to
//! the trailing boundary marker). Not a full RFC 2046 parser -- deliberately
//! byte-for-byte equivalent to what jarvis already shipped and had verified
//! working, not a "more correct" reimplementation.

pub fn boundary_from_content_type(content_type: &str) -> Option<String> {
    content_type.split("boundary=").nth(1).map(|b| b.trim().to_string())
}

/// Finds the first part containing `filename` (or, for legacy callers, any
/// marker byte string) in its headers and returns its raw body bytes.
pub fn extract_file_part(body: &[u8], boundary: &str, marker: &[u8]) -> Option<Vec<u8>> {
    let delim = format!("--{boundary}");
    let delim = delim.as_bytes();
    for part in split_on(body, delim) {
        if contains(part, marker)
            && let Some(idx) = find(part, b"\r\n\r\n") {
                let mut content = &part[idx + 4..];
                // rstrip(b"\r\n--") equivalent: trim a trailing CRLF and/or
                // the leading "--" of the next boundary marker if present.
                while content.ends_with(b"-") || content.ends_with(b"\r") || content.ends_with(b"\n") {
                    content = &content[..content.len() - 1];
                }
                return Some(content.to_vec());
            }
    }
    None
}

fn split_on<'a>(haystack: &'a [u8], needle: &[u8]) -> Vec<&'a [u8]> {
    let mut parts = Vec::new();
    let mut start = 0;
    while let Some(pos) = find(&haystack[start..], needle) {
        parts.push(&haystack[start..start + pos]);
        start += pos + needle.len();
    }
    parts.push(&haystack[start..]);
    parts
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    find(haystack, needle).is_some()
}

/// Extracts `filename="..."` from a Content-Disposition-bearing part.
pub fn extract_filename(part: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(part);
    let key = "filename=\"";
    let start = text.find(key)? + key.len();
    let end = text[start..].find('"')? + start;
    Some(text[start..end].to_string())
}
