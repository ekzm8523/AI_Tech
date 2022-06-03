import zipfile

if __name__ == '__main__':
    path_to_zip_file = 'Data/AIHub-data/03.Clue_.zip'
    directory_to_extract_to = '압축 풀고 저장할 경로'

    import zipfile

    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)