use crate::languages::{Language, get_languages};
use code_splitter::Chunk;
use polars::prelude::*;
use std::fs::ReadDir;
use std::io::{BufRead, Read, Seek};
use std::path::{Path, PathBuf};
use std::{fs, io};

struct FileContent {
    lines: Vec<String>,
    chunks: Vec<Chunk>,
}

#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    file_path: String,
    file_name: String,
    start_line: u64,
    end_line: u64,
    text: Option<String>,
    size: u64,
}

struct CodeFileSplitter {
    directories: Vec<PathBuf>,
    entries: Option<ReadDir>,
    chunks: Vec<ChunkMetadata>,
}

impl From<String> for CodeFileSplitter {
    fn from(path: String) -> Self {
        CodeFileSplitter {
            directories: vec![PathBuf::from(path)],
            entries: None,
            chunks: vec![],
        }
    }
}

impl Iterator for CodeFileSplitter {
    type Item = ChunkMetadata;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.chunks.is_empty() || !self.directories.is_empty() || self.entries.is_some() {
            // Process chunks.
            if !self.chunks.is_empty() {
                return Some(self.chunks.remove(0));
            }

            // Process directory entries.
            while let Some(read_dir) = &mut self.entries {
                match read_dir.next() {
                    Some(Ok(entry)) => {
                        let path = entry.path();
                        if let Ok(metadata) = entry.metadata() {
                            if metadata.is_dir() {
                                self.directories.push(path.clone());
                                continue;
                            } else {
                                let maybe_chunks = CodeFileSplitter::process_file(&path);
                                if let Some(mut chunks) = maybe_chunks {
                                    self.chunks.append(&mut chunks);
                                }
                            }
                        }
                    }
                    None => {
                        // All done in this directory.
                        self.entries = None;
                        break;
                    }
                    _ => {}
                }
            }

            // Process directories.
            while let Some(directory) = self.directories.pop() {
                if let Ok(entries) = fs::read_dir(&directory) {
                    self.entries = Some(entries);
                }
            }
        }
        None
    }
}

impl CodeFileSplitter {
    fn split_file(path: &Path, language: &Language) -> Result<FileContent, code_splitter::Error> {
        let file = fs::File::open(path)?;

        let mut reader = io::BufReader::new(file);

        let lines: Vec<String> = reader.by_ref().lines().collect::<Result<_, _>>()?;

        let mut contents = String::new();
        reader.rewind()?;
        reader.read_to_string(&mut contents)?;
        let chunks = language.splitter.split(contents.as_bytes())?;

        Ok(FileContent { lines, chunks })
    }

    fn process_file(path: &Path) -> Option<Vec<ChunkMetadata>> {
        let maybe_extension = path.extension().unwrap().to_str().map(|x| x.to_string());

        if let Some(extension) = maybe_extension {
            let maybe_processed_content = get_languages()
                .iter()
                .filter(|x| x.extensions.contains(&extension))
                .map(|y| CodeFileSplitter::split_file(path, y))
                .next()?
                .map_err(|e| eprintln!("Failed to process file: {e}"));

            if maybe_processed_content.is_ok() {
                let processed_content = maybe_processed_content.unwrap();
                let mut chunks = Vec::new();
                for chunk in processed_content.chunks {
                    chunks.push(ChunkMetadata {
                        file_path: path.to_str().unwrap_or_default().to_string(),
                        file_name: path
                            .file_name()
                            .unwrap()
                            .to_str()
                            .unwrap_or_default()
                            .to_string(),
                        start_line: chunk.range.start_point.row as u64,
                        end_line: chunk.range.end_point.row as u64,
                        text: Some(
                            processed_content.lines
                                [chunk.range.start_point.row..chunk.range.end_point.row]
                                .join("\n"),
                        )
                        .filter(|x| !x.is_empty()),
                        size: chunk.size as u64,
                    });
                }
                return Some(chunks);
            }
        }
        None
    }
}

///
///
/// # Arguments
///
/// * `input_dir_path`:
/// * `output_file_uri`:
///
/// returns: Result<(), String>
///
/// # Examples
///
/// ```
///
/// ```
pub fn find_and_split(input_dir_path: String, output_file_uri: String) -> Result<(), String> {
    let splitter = CodeFileSplitter::from(input_dir_path);
    let mut output_file = fs::File::create(output_file_uri).map_err(|e| e.to_string())?;

    let mut file_paths: Vec<String> = Vec::new();
    let mut file_names: Vec<String> = Vec::new();
    let mut start_lines: Vec<u64> = Vec::new();
    let mut end_lines: Vec<u64> = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    let mut sizes: Vec<u64> = Vec::new();

    for chunk in splitter {
        if chunk.text.is_some() {
            file_paths.push(chunk.file_path);
            file_names.push(chunk.file_name);
            start_lines.push(chunk.start_line);
            end_lines.push(chunk.end_line);
            texts.push(chunk.text.unwrap());
            sizes.push(chunk.size);
        }
    }
    let mut dataframe = df!(
        "file_path" => file_paths,
        "file_name" => file_names,
        "start_line" => start_lines,
        "end_line" => end_lines,
        "text" => texts,
        "size" => sizes,
    )
    .map_err(|x| x.to_string())?;

    ParquetWriter::new(&mut output_file)
        .finish(&mut dataframe)
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::path::{Path, PathBuf};

    // Helper function for creating temporary test files
    fn create_temp_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let file_path = dir.join(name);
        let mut file = File::create(&file_path).expect("Failed to create test file");
        writeln!(file, "{content}").expect("Failed to write to test file");
        file_path
    }

    #[test]
    fn test_split_file() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let temp_file_path = create_temp_file(
            temp_dir.path(),
            "test_file.rs",
            r#"
            fn main() {
                println!("Hello, world!");
            }
            "#,
        );

        let language = get_languages().iter().find(|x| x.name.eq("rust")).unwrap();
        let result = CodeFileSplitter::split_file(&temp_file_path, language);

        // Assert that the split succeeded and returned the correct structure
        assert!(result.is_ok());

        let file_content = result.unwrap();
        assert!(
            !file_content.lines.is_empty(),
            "File lines should not be empty"
        );
        assert!(
            !file_content.chunks.is_empty(),
            "Chunks should not be empty"
        );
    }

    #[test]
    fn test_process_file() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let temp_file_path = create_temp_file(
            temp_dir.path(),
            "test_file.rs",
            r#"
            fn main() {
                println!("Processing file test");
            }
            "#,
        );

        let result = CodeFileSplitter::process_file(&temp_file_path);

        // Assert processing results
        assert!(result.is_some(), "Processing result should not be None");
        let chunks = result.unwrap();
        // Example: Check that chunks have metadata
        assert!(!chunks.is_empty(), "Chunks should not be empty");

        // Validate chunk contents
        let chunk = &chunks[0];
        assert_eq!(chunk.file_path, temp_file_path.to_string_lossy());
        assert!(
            chunk.text.clone().unwrap().contains("fn main()"),
            "First chunk should contain the main function"
        );
        assert!(chunk.size > 0, "Chunk size should be greater than 0");
    }

    #[test]
    fn test_code_file_splitter_iterator() {
        let root_temp_dir = tempfile::tempdir().expect("Failed to create root temp directory");
        let first_level_temp_dir = tempfile::tempdir_in(root_temp_dir.path())
            .expect("Failed to create first level temp directory");
        let second_level_temp_dir = tempfile::tempdir_in(first_level_temp_dir.path())
            .expect("Failed to create second level temp directory");

        let temp_file_path1 = create_temp_file(
            first_level_temp_dir.path(),
            "test_file_first.rs",
            r#"
            fn foo() {
                println!("File 1");
            }
            "#,
        );
        let temp_file_path2 = create_temp_file(
            second_level_temp_dir.path(),
            "test_file_second.rs",
            r#"
            fn bar() {
                println!("File 2");
            }
            "#,
        );

        let splitter = CodeFileSplitter::from(root_temp_dir.path().to_str().unwrap().to_string());

        let processed_chunks: Vec<_> = splitter.collect();

        assert_eq!(processed_chunks.len(), 2);
        assert_eq!(
            processed_chunks[0].file_path,
            temp_file_path1.to_string_lossy()
        );
        assert_eq!(
            processed_chunks[1].file_path,
            temp_file_path2.to_string_lossy()
        );
    }

    #[test]
    fn test_find_and_split() {
        let root_temp_dir = tempfile::tempdir().expect("Failed to create root temp directory");
        let first_level_temp_dir = tempfile::tempdir_in(root_temp_dir.path())
            .expect("Failed to create first level temp directory");
        let second_level_temp_dir = tempfile::tempdir_in(first_level_temp_dir.path())
            .expect("Failed to create second level temp directory");

        let temp_file_content_first = r#"
            fn foo() {
                println!("File 1");
            }
            "#;
        let temp_file_path_first = create_temp_file(
            first_level_temp_dir.path(),
            "test_file_first.rs",
            temp_file_content_first,
        );

        let temp_file_content_second = format!(
            "{}\n{}\n{}",
            r#"
            fn bar() {
                println!("File 2");
            "#,
            r#"
                let mut file_paths: Vec<String> = Vec::new();
                let mut file_names: Vec<String> = Vec::new();
                let mut start_lines: Vec<u64> = Vec::new();
                let mut end_lines: Vec<u64> = Vec::new();
                let mut texts: Vec<Option<String>> = Vec::new();
                let mut sizes: Vec<u64> = Vec::new();
            "#
            .repeat(40),
            r#" 
            }
            "#
        );
        let temp_file_path_second = create_temp_file(
            second_level_temp_dir.path(),
            "test_file_second.rs",
            temp_file_content_second.as_str(),
        );

        let temp_file_paths = [
            temp_file_path_first.as_path().to_str().unwrap(),
            temp_file_path_second.as_path().to_str().unwrap(),
        ];
        let temp_file_names = [
            temp_file_path_first.file_name().unwrap().to_str().unwrap(),
            temp_file_path_second.file_name().unwrap().to_str().unwrap(),
        ];

        let output_file_uri = root_temp_dir.path().join("output_file.parquet");

        let _ = find_and_split(
            root_temp_dir.path().to_str().unwrap().to_string(),
            output_file_uri.to_str().unwrap().to_string(),
        );

        assert!(output_file_uri.exists());
        let dataframe = LazyFrame::scan_parquet(output_file_uri, Default::default())
            .unwrap()
            .collect()
            .unwrap();
        assert!(!dataframe.is_empty());
        assert_eq!(
            dataframe.get_column_names(),
            &[
                "file_path",
                "file_name",
                "start_line",
                "end_line",
                "text",
                "size"
            ]
        );
        assert_eq!(dataframe.shape(), (4, 6));
        assert!(
            dataframe
                .column("file_path")
                .unwrap()
                .str()
                .unwrap()
                .iter()
                .map(|x| x.unwrap())
                .all(|x| temp_file_paths.contains(&x))
        );
        assert!(
            dataframe
                .column("file_name")
                .unwrap()
                .str()
                .unwrap()
                .iter()
                .map(|x| x.unwrap())
                .all(|x| temp_file_names.contains(&x))
        );
        assert!(
            dataframe
                .column("text")
                .unwrap()
                .str()
                .unwrap()
                .iter()
                .map(|x| x.unwrap())
                .all(
                    |x| temp_file_content_first.contains(x) || temp_file_content_second.contains(x)
                )
        );
    }
}
