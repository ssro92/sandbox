import pandas as pd
import psycopg2
from sqlalchemy import create_engine, inspect
from typing import Dict, List, Any, Tuple, Union
import numpy as np

class PostgreSQLQueryGenerator:
    """
    판다스 데이터프레임을 분석하여 PostgreSQL SELECT 쿼리를 생성하는 클래스
    """
    
    def __init__(self, df: pd.DataFrame, table_name: str, primary_keys: List[str]):
        """
        Args:
            df: 분석할 판다스 데이터프레임
            table_name: 대상 PostgreSQL 테이블명
            primary_keys: 다중 Primary Key 컬럼 리스트
        """
        self.df = df
        self.table_name = table_name
        self.primary_keys = primary_keys
        self.column_info = self._analyze_dataframe()
    
    def _analyze_dataframe(self) -> Dict[str, Dict[str, Any]]:
        """데이터프레임의 컬럼과 데이터 타입을 분석합니다."""
        column_info = {}
        
        for column in self.df.columns:
            dtype = self.df[column].dtype
            
            # 판다스 데이터 타입을 PostgreSQL 데이터 타입으로 매핑
            if pd.api.types.is_integer_dtype(dtype):
                pg_type = 'INTEGER'
            elif pd.api.types.is_float_dtype(dtype):
                pg_type = 'NUMERIC'
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                pg_type = 'TIMESTAMP'
            elif pd.api.types.is_bool_dtype(dtype):
                pg_type = 'BOOLEAN'
            else:
                pg_type = 'TEXT'
            
            column_info[column] = {
                'pg_type': pg_type,
                'pandas_type': str(dtype),
                'is_timestamp': pd.api.types.is_datetime64_any_dtype(dtype)
            }
        
        return column_info
    
    def _format_select_columns(self) -> str:
        """SELECT 절의 컬럼들을 포맷팅합니다. timestamp 컬럼은 to_char로 변환합니다."""
        select_columns = []
        
        for column in self.df.columns:
            if self.column_info[column]['is_timestamp']:
                # timestamp 컬럼을 문자열로 변환
                formatted_column = f"to_char({column}, 'YYYY-MM-DD HH24:MI:SS') AS {column}"
                select_columns.append(formatted_column)
            else:
                select_columns.append(column)
        
        return ', '.join(select_columns)
    
    def _format_pk_values(self, pk_values: List[Tuple]) -> str:
        """다중 PK 값들을 IN 절 형태로 포맷팅합니다."""
        if not pk_values:
            return ""
        
        formatted_values = []
        for pk_tuple in pk_values:
            # 각 PK 값을 적절한 형태로 포맷팅
            formatted_tuple_values = []
            for i, value in enumerate(pk_tuple):
                pk_column = self.primary_keys[i]
                column_type = self.column_info.get(pk_column, {}).get('pg_type', 'TEXT')
                
                if pd.isna(value) or value is None:
                    formatted_tuple_values.append('NULL')
                elif column_type in ['INTEGER', 'NUMERIC']:
                    formatted_tuple_values.append(str(value))
                elif column_type == 'BOOLEAN':
                    formatted_tuple_values.append(str(value).upper())
                elif column_type == 'TIMESTAMP':
                    # timestamp 값을 문자열로 변환
                    if isinstance(value, pd.Timestamp):
                        formatted_tuple_values.append(f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'")
                    else:
                        formatted_tuple_values.append(f"'{value}'")
                else:
                    # TEXT 타입
                    formatted_tuple_values.append(f"'{value}'")
            
            formatted_values.append(f"({', '.join(formatted_tuple_values)})")
        
        return ', '.join(formatted_values)
    
    def generate_query_with_multi_pk(self, pk_values: List[Tuple] = None) -> str:
        """
        다중 PK 조건을 포함한 SELECT 쿼리를 생성합니다.
        
        Args:
            pk_values: PK 값들의 튜플 리스트. None인 경우 데이터프레임에서 추출
        
        Returns:
            생성된 SELECT 쿼리 문자열
        """
        # SELECT 절 생성 (timestamp 컬럼 포함)
        select_columns = self._format_select_columns()
        
        # 기본 쿼리 구조
        query = f"SELECT {select_columns} FROM {self.table_name} WHERE 1=1"
        
        # PK 값들 처리
        if pk_values is None:
            # 데이터프레임에서 PK 값들 추출
            pk_values = self._extract_pk_values_from_df()
        
        if pk_values:
            # 다중 PK IN 절 추가
            pk_columns = ', '.join(self.primary_keys)
            formatted_pk_values = self._format_pk_values(pk_values)
            
            if formatted_pk_values:
                query += f" AND ({pk_columns}) IN ({formatted_pk_values})"
        
        return query
    
    def _extract_pk_values_from_df(self) -> List[Tuple]:
        """데이터프레임에서 PK 값들을 추출합니다."""
        pk_values = []
        
        # 모든 PK 컬럼이 데이터프레임에 존재하는지 확인
        missing_pk_columns = [pk for pk in self.primary_keys if pk not in self.df.columns]
        if missing_pk_columns:
            print(f"경고: 다음 PK 컬럼들이 데이터프레임에 없습니다: {missing_pk_columns}")
            return pk_values
        
        # 각 행에서 PK 값들을 튜플로 추출
        for _, row in self.df.iterrows():
            pk_tuple = tuple(row[pk] for pk in self.primary_keys)
            pk_values.append(pk_tuple)
        
        # 중복 제거
        pk_values = list(set(pk_values))
        
        return pk_values
    
    def generate_query_with_filters(self, additional_filters: Dict[str, Any] = None) -> str:
        """
        추가 필터 조건을 포함한 쿼리를 생성합니다.
        
        Args:
            additional_filters: 추가 필터 조건 딕셔너리
        
        Returns:
            필터가 적용된 SELECT 쿼리 문자열
        """
        # 기본 다중 PK 쿼리 생성
        query = self.generate_query_with_multi_pk()
        
        if additional_filters:
            where_conditions = []
            
            for column, value in additional_filters.items():
                if column in self.df.columns:
                    column_type = self.column_info[column]['pg_type']
                    
                    if column_type in ['TEXT']:
                        where_conditions.append(f"{column} = '{value}'")
                    elif column_type in ['INTEGER', 'NUMERIC']:
                        where_conditions.append(f"{column} = {value}")
                    elif column_type == 'BOOLEAN':
                        where_conditions.append(f"{column} = {str(value).upper()}")
                    elif column_type == 'TIMESTAMP':
                        if isinstance(value, pd.Timestamp):
                            where_conditions.append(f"{column} = '{value.strftime('%Y-%m-%d %H:%M:%S')}'")
                        else:
                            where_conditions.append(f"{column} = '{value}'")
            
            if where_conditions:
                query += f" AND {' AND '.join(where_conditions)}"
        
        return query

def execute_query_with_connection(query: str, connection_params: Dict[str, str]) -> pd.DataFrame:
    """
    생성된 쿼리를 실행하여 결과를 데이터프레임으로 반환합니다.
    
    Args:
        query: 실행할 SQL 쿼리
        connection_params: PostgreSQL 연결 파라미터
    
    Returns:
        쿼리 결과 데이터프레임
    """
    engine = create_engine(
        f"postgresql://{connection_params['user']}:{connection_params['password']}@"
        f"{connection_params['host']}:{connection_params['port']}/{connection_params['database']}"
    )
    
    try:
        result_df = pd.read_sql_query(query, engine)
        return result_df
    
    except Exception as e:
        print(f"쿼리 실행 중 오류 발생: {e}")
        return pd.DataFrame()
    
    finally:
        engine.dispose()

# 사용 예시
def main():
    # 샘플 데이터프레임 생성 (다중 PK와 timestamp 포함)
    sample_data = {
        'pk1': [1, 2, 3, 4, 5],
        'pk2': ['A', 'B', 'C', 'D', 'E'],
        'pk3': [100, 200, 300, 400, 500],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000.0, 60000.0, 70000.0, 55000.0, 65000.0],
        'is_active': [True, True, False, True, True],
        'created_at': pd.to_datetime([
            '2023-01-01 10:30:45', 
            '2023-01-02 14:20:30', 
            '2023-01-03 09:15:22', 
            '2023-01-04 16:45:10', 
            '2023-01-05 11:30:55'
        ]),
        'updated_at': pd.to_datetime([
            '2023-06-01 08:20:15', 
            '2023-06-02 13:45:30', 
            '2023-06-03 17:30:45', 
            '2023-06-04 12:15:20', 
            '2023-06-05 09:50:35'
        ])
    }
    
    df = pd.DataFrame(sample_data)
    
    # 다중 PK 정의
    primary_keys = ['pk1', 'pk2', 'pk3']
    
    # 쿼리 생성기 초기화
    query_generator = PostgreSQLQueryGenerator(df, 'employees', primary_keys)
    
    # 1. 데이터프레임의 모든 PK 값으로 쿼리 생성
    query1 = query_generator.generate_query_with_multi_pk()
    print("1. 데이터프레임 기반 다중 PK 쿼리:")
    print(query1)
    print()
    
    # 2. 특정 PK 값들로 쿼리 생성
    specific_pk_values = [
        (1, 'A', 100),
        (2, 'B', 200),
        (3, 'C', 300)
    ]
    query2 = query_generator.generate_query_with_multi_pk(specific_pk_values)
    print("2. 특정 PK 값들로 생성된 쿼리:")
    print(query2)
    print()
    
    # 3. 추가 필터 조건을 포함한 쿼리
    additional_filters = {'is_active': True, 'age': 30}
    query3 = query_generator.generate_query_with_filters(additional_filters)
    print("3. 추가 필터 조건 포함 쿼리:")
    print(query3)
    print()
    
    # 컬럼 정보 출력
    print("4. 컬럼 정보:")
    for col, info in query_generator.column_info.items():
        print(f"  {col}: {info['pg_type']} (timestamp: {info['is_timestamp']})")

if __name__ == "__main__":
    main()
