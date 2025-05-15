use anyhow::anyhow;
use fastembed::{Embedding, EmbeddingModel, InitOptions, TextEmbedding};
#[cfg(target_os = "macos")]
use ort::execution_providers::{
    CoreMLExecutionProvider, ExecutionProvider, ExecutionProviderDispatch,
};
use polars::datatypes::DataType;
use polars::datatypes::DataType::List;
use polars::prelude::{Column, GetOutput, LazyFrame, ParquetWriter, col};
use polars::series::Series;
use std::fs;
use std::sync::OnceLock;

static TEXT_EMBEDDING_MODEL: OnceLock<TextEmbedding> = OnceLock::new();

#[cfg(target_os = "macos")]
fn register_provider() -> anyhow::Result<ExecutionProviderDispatch> {
    let coreml = CoreMLExecutionProvider::default();
    if !coreml.is_available()? {
        return Err(anyhow!("CoreML provider is not available".to_string()));
    }

    Ok(coreml.with_subgraphs().build())
}

#[cfg(target_os = "windows")]
fn register_provider() -> Result<ExecutionProviderDispatch, String> {
    todo!()
}

fn get_text_embedding_model() -> anyhow::Result<&'static TextEmbedding> {
    let execution_provider = register_provider()?;
    TEXT_EMBEDDING_MODEL.get_or_try_init(|| {
        TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                .with_execution_providers(vec![execution_provider]),
        )
    })
}

pub fn create_embeddings_from_file(
    input_file_uri: String,
    output_file_uri: String,
) -> anyhow::Result<()> {
    let mut output_file = fs::File::create(output_file_uri)?;

    // Read a dataframe from a file.
    let dataframe = LazyFrame::scan_parquet(input_file_uri, Default::default())?;

    // Create a model.
    let model: &TextEmbedding = get_text_embedding_model()?;

    let mut dataframe_plus_embeddings = dataframe
        .with_column(col("text").alias("embedding").map_list(
            move |x| {
                let as_string_chunked = x.as_series().unwrap().str()?;
                let embeddings: Vec<Series> = as_string_chunked
                    .into_iter()
                    .flat_map(|y| {
                        model
                            .embed::<String>(vec![y.unwrap().into()], Some(32))
                            .unwrap()
                    })
                    .map(|z| z.into_iter().collect::<Series>())
                    .collect();

                //let series = Series::new("embeddings".into(), &embeddings);
                Ok(Some(Column::new("embedding".into(), &embeddings)))
            },
            GetOutput::from_type(List(Box::new(DataType::Float32))),
        ))
        .collect()?;

    ParquetWriter::new(&mut output_file).finish(&mut dataframe_plus_embeddings)?;
    Ok(())
}

pub fn create_embeddings_from_string(input_string: String) -> anyhow::Result<Embedding> {
    let model: &TextEmbedding = get_text_embedding_model()?;
    let embedding = model.embed::<String>(vec![input_string], Some(32))?;
    Ok(embedding[0].clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::split::find_and_split;
    use std::fs::File;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use tempfile::TempDir;

    fn create_temp_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let file_path = dir.join(name);
        let mut file = File::create(&file_path).expect("Failed to create test file");
        writeln!(file, "{}", content).expect("Failed to write to test file");
        file_path
    }

    fn create_split_parquet_file(output_dir_path: &TempDir) -> PathBuf {
        let first_level_temp_dir = tempfile::tempdir_in(output_dir_path.path())
            .expect("Failed to create first level temp directory");
        let second_level_temp_dir = tempfile::tempdir_in(first_level_temp_dir.path())
            .expect("Failed to create second level temp directory");

        let temp_file_content_first = r#"
            fn foo() {
                println!("File 1");
            }
            "#;
        let _ = create_temp_file(
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
        let _ = create_temp_file(
            second_level_temp_dir.path(),
            "test_file_second.rs",
            temp_file_content_second.as_str(),
        );

        let output_file_uri = output_dir_path.path().join("split_file.parquet");

        assert!(
            find_and_split(
                output_dir_path.path().to_str().unwrap().to_string(),
                output_file_uri.to_str().unwrap().to_string(),
            )
            .is_ok()
        );

        output_file_uri
    }

    #[test]
    fn test_create_embeddings() {
        let root_temp_dir = tempfile::tempdir().expect("Failed to create root temp directory");
        let input_file_uri = create_split_parquet_file(&root_temp_dir);
        let output_file_uri = root_temp_dir.path().join("embed_file.parquet");

        // Run the function to test
        let result = create_embeddings_from_file(
            input_file_uri.to_str().unwrap().to_string(),
            output_file_uri.to_str().unwrap().to_string(),
        );

        // Assert the function executed successfully
        assert!(
            result.is_ok(),
            "Failed to create embeddings: {:?}",
            result.err()
        );

        // Read the output file to verify it contains embeddings
        let output_df = LazyFrame::scan_parquet(output_file_uri, Default::default())
            .unwrap()
            .collect()
            .unwrap();

        // Verify the output DataFrame has both text and embedding columns
        assert!(output_df.schema().contains("text"));
        assert!(output_df.schema().contains("embedding"));

        // Verify embeddings are non-empty
        let embedding_col = output_df.column("embedding").unwrap();
        for i in 0..embedding_col.len() {
            let embedding = embedding_col.get(i).unwrap();
            // Check that the embedding exists and is not empty
            assert!(!embedding.is_null());
        }
    }
}
