use std::fs::File;
use lancedb::connect;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub async fn index(input_file_uri: String, db_file_uri: String) -> anyhow::Result<()> {
    let file = File::open(input_file_uri)?;

    let reader_builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let record_batch_reader = reader_builder.build()?;

    let db_connection = connect(db_file_uri.as_str()).execute().await?;

    // Check if the table exists.
    let maybe_table = db_connection.open_table("codebases").execute().await;
    if maybe_table.is_err() {
        // Table does not exist, so let's create it and load it with data.
        db_connection
            .create_table("codebases", record_batch_reader)
            .execute()
            .await?;
    } else {
        // Table exists already; add the new records.
        maybe_table?.add(record_batch_reader).execute().await?;
    };

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lancedb::query::ExecutableQuery;
    use polars::df;
    use polars::prelude::*;
    use lancedb::arrow::IntoPolars;
    use tempfile::TempDir;

    async fn create_test_parquet() -> (TempDir, String) {
        // Create a temporary directory that will be automatically cleaned up
        let temp_dir = TempDir::new().unwrap();
        let parquet_path = temp_dir.path().join("test.parquet");

        // Create a simple DataFrame
        let mut df = df!(
            "id" => &[1, 2, 3],
            "name" => &["test1", "test2", "test3"],
            "value" => &[10.0, 20.0, 30.0]
        )
        .unwrap();

        // Write DataFrame to parquet
        let mut file = std::fs::File::create(&parquet_path).unwrap();
        ParquetWriter::new(&mut file).finish(&mut df).unwrap();

        (temp_dir, parquet_path.to_string_lossy().to_string())
    }

    #[tokio::test]
    async fn test_index_creates_new_table() -> anyhow::Result<()> {
        // Setup
        let (temp_dir, parquet_path) = create_test_parquet().await;
        let db_path = temp_dir.path().join("test.db");
        let db_path_str = db_path.to_string_lossy().to_string();

        // Execute
        index(parquet_path, db_path_str.clone()).await?;

        // Verify
        let db = connect(&db_path_str).execute().await?;
        let table = db.open_table("codebases").execute().await?;

        // Convert to DataFrame for easy verification
        let stream = table.query().execute().await?;
        let df = stream.into_polars().await?;

        assert_eq!(df.shape().0, 3);
        assert_eq!(df.shape().1, 3);

        // Check column names
        let column_names: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|&s| s.to_string())
            .collect();
        assert!(column_names.contains(&"id".to_string()));
        assert!(column_names.contains(&"name".to_string()));
        assert!(column_names.contains(&"value".to_string()));

        Ok(())
    }

    #[tokio::test]
    async fn test_index_adds_to_existing_table() -> Result<(), Box<dyn std::error::Error>> {
        // Setup
        let (temp_dir, parquet_path) = create_test_parquet().await;
        let db_path = temp_dir.path().join("test.db");
        let db_path_str = db_path.to_string_lossy().to_string();

        // First insertion
        index(parquet_path.clone(), db_path_str.clone()).await?;

        // Second insertion
        index(parquet_path, db_path_str.clone()).await?;

        // Verify
        let db = connect(&db_path_str).execute().await?;
        let table = db.open_table("codebases").execute().await?;

        let df = table.query().execute().await?.into_polars().await?;

        // Should have 6 rows (3 from each insertion)
        assert_eq!(df.shape().0, 6);
        assert_eq!(df.shape().1, 3);

        Ok(())
    }

    #[tokio::test]
    async fn test_index_with_invalid_parquet() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let invalid_parquet_path = temp_dir.path().join("nonexistent.parquet");
        let db_path = temp_dir.path().join("test.db");

        let result = index(
            invalid_parquet_path.to_string_lossy().to_string(),
            db_path.to_string_lossy().to_string(),
        )
        .await;

        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_index_with_invalid_db_path() -> Result<(), Box<dyn std::error::Error>> {
        let (_temp_dir, parquet_path) = create_test_parquet().await;
        let invalid_db_path = "/nonexistent/path/that/should/fail/test.db";

        let result = index(parquet_path, invalid_db_path.to_string()).await;

        assert!(result.is_err());
        Ok(())
    }
}
