import oracledb
from .settings import Settings
from typing import List


class ODBType:
    """Clase que define los tipos de datos de Oracle"""
    VARCHAR2 = "VARCHAR2"
    DECIMAL = "NUMBER"
    SYS_REFCURSOR = "CURSOR"
    PSQLCODE = "PSQLCODE"
    NUMBER = "NUMBER"
    DATE = "DATE"
    INT = "INTEGER"
    VARCHARLIST = "ARRAYVARCHAR2"


class ODBFunctionType:
    """Clase que define los tipos de funciones de Oracle"""
    NUMBER = "INT"
    STRING = "STRING"
    CURSOR = "CURSOR"


def convert_camel_case(texto: str) -> str:
    """
        Convierte una cadena en formato snake_case a camelCase.

        Args:
            texto (str): La cadena en formato snake_case a convertir.

        Returns:
            str: La cadena convertida en formato camelCase.
    """
    palabras = texto.split("_")
    camel_case = '_'.join(palabra.capitalize() for i, palabra in enumerate(palabras))
    return camel_case


def process_sys_refcursor(sys_refcursor):
    """
    Procesa un objeto SYS_REFCURSOR y devuelve una lista de diccionarios.

    Args:
        sys_refcursor: El objeto SYS_REFCURSOR a procesar.

    Returns:
        List[dict]: Una lista de diccionarios que representa las filas en el SYS_REFCURSOR.
    """
    cursor_results = []
    for row in sys_refcursor:
        row_dict = {}
        for i, col in enumerate(sys_refcursor.description):
            col_name = col[0]
            col_value = row[i]
            if isinstance(col_value, oracledb.LOB):
                col_value = col_value.read()
            col_name_camel_case = convert_camel_case(col_name)
            row_dict[col_name_camel_case] = col_value

        cursor_results.append(row_dict)
    return cursor_results


def get_dynamic_values(dynamic_var_names, dynamic_vars, dynamic_var_types):
    """
        Obtiene los valores dinámicos de las variables en base a sus nombres, valores y tipos.

        Args:
            dynamic_var_names (list): Lista de nombres de variables dinámicas.
            dynamic_vars (list): Lista de valores de variables dinámicas.
            dynamic_var_types (list): Lista de tipos de variables dinámicas.

        Returns:
            dict: Un diccionario que contiene los valores dinámicos indexados por nombre de variable.
    """
    dynamic_values = {}
    for name, dynamic_var, param_type in zip(dynamic_var_names, dynamic_vars, dynamic_var_types):
        if param_type == ODBType.SYS_REFCURSOR:
            value = []
            cursor_var = dynamic_var.getvalue()
            if cursor_var is not None:
                for row in cursor_var:
                    row_dict = dict(zip([str(d[0]) for d in cursor_var.description], row))
                    converted_row_dict = {convert_camel_case(k): v for k, v in row_dict.items()}
                    value.append(converted_row_dict)
        elif param_type == ODBType.PSQLCODE:
            value = dynamic_var.getvalue()
        else:
            value = dynamic_var
        dynamic_values[name] = value
    return dynamic_values


class Oradb:
    """
        Clase que proporciona métodos para conectar y administrar una conexión a una base de datos Oracle.

        Attributes:
            db_user (str): Nombre de usuario para la conexión a la base de datos.
            db_pass (str): Contraseña para la conexión a la base de datos.
            db_dns (str): DSN (Data Source Name) para la conexión a la base de datos.
            min_conns (int): Número mínimo de conexiones en el pool.
            max_conns (int): Número máximo de conexiones en el pool.
            incr_conns (int): Incremento de conexiones en el pool.
            pool (obj): Pool de conexiones Oracle.
            connection (obj): Conexión activa a la base de datos.
    """

    def __init__(self):
        """
            Inicializa una instancia de OracleDBConnector y configura los atributos necesarios para la conexión.
        """
        settings = Settings()
        self.db_user = settings.db_user
        self.db_pass = settings.db_pass
        self.db_dns = settings.db_dns
        self.min_conns = settings.min_conns
        self.max_conns = settings.max_conns
        self.incr_conns = settings.incr_conns
        self.pool = settings.pool
        self.connection = None
        self.transaction = False

    def create_pool(self):
        """
            Crea un pool de conexiones Oracle.

            Returns:
                None

            Raises:
                ConnectionError: Si ocurre un error al crear el pool de conexiones.
        """
        try:
            self.pool = oracledb.create_pool(
                user=self.db_user,
                password=self.db_pass,
                dsn=self.db_dns,
                min=self.min_conns,
                max=self.max_conns,
                increment=self.incr_conns
            )
        except oracledb.DatabaseError as e:
            raise ConnectionError("Error al crear el pool de conexiones: " + str(e))

    def close_connection(self):
        """
            Cierra la conexión activa.

            Returns:
                None
        """
        if self.connection is not None:
            self.connection.close()
            self.connection = None
            self.transaction = False

    def connect(self,transaction = False):
        """
           Establece una conexión utilizando el pool de conexiones.

           Returns:
               None

           Raises:
               ConnectionError: Si ocurre un error al establecer la conexión.
       """
        try:
            self.connection = self.pool.acquire()
            self.transaction = transaction
        except oracledb.DatabaseError as e:
            raise ConnectionError("Error al establecer la conexión: " + str(e))
        

    # 07-2023, rcastro
    def begin_transaction(self):
        """
            Inicia una transaccion de base de datos, esto abre la conexion a la base
            de datos y la mantiene asi hasta que se ejecute un commit o rollback

            Returns:
                None 
        """
        if self.connection is None:
            self.create_pool()
            self.connect(transaction=True)
            self.connection.begin()


    # 07-2023, rcastro
    def commit_transaction(self):
        """
            Confirma las transacciones ejecutadas en la base de datos y cierra la conexion

            Returns:
                None 
        """
        if self.connection is not None:
            self.connection.commit()
            self.close_connection()
            self.disconnect()


    # 07-2023, rcastro
    def rollback(self):
        """
            Revierte las transacciones ejecutadas en la base de datos y cierra la conexion

            Returns:
                None 
        """
        if self.connection is not None:
            self.connection.rollback()
            self.close_connection()
            self.disconnect()
            

    def execute_procedure(self, procedure_name: str, params: List[dict]):
        """
            Ejecuta un procedimiento almacenado en Oracle con los parámetros proporcionados.

            Args:
                procedure_name (str): Nombre del procedimiento almacenado a ejecutar.
                params (List[dict]): Lista de diccionarios que contienen información de los parámetros.
                    Cada diccionario debe contener las siguientes claves:
                    - "nombre": Nombre del parámetro.
                    - "valor": Valor del parámetro.
                    - "tipo": Tipo de dato del parámetro (debe ser uno de los valores definidos en la clase ODBType).
                    - "direccion": Dirección del parámetro (debe ser uno de los valores: "IN", "OUT", "IN/OUT").

            Returns:
                Union[dict, None]: Un diccionario con los valores de los parámetros de salida o None si ocurre un error.

            Raises:
                ConnectionError: Si ocurre un error al crear o establecer la conexión con la base de datos.
        """
        try:            
            if self.transaction is False:
                self.create_pool()
                self.connect()
            with self.connection.cursor() as cursor:
                send_params = []
                dynamic_vars = []
                dynamic_var_names = []
                dynamic_var_types = []
                ora_psqlcode = {}
                for param in params:
                    param_name = param["nombre"]
                    param_value = param["valor"]
                    param_type = param["tipo"]
                    param_direction = param["direccion"]

                    if param_type == ODBType.DATE:
                        param_value = oracledb.Timestamp.from_datetime(param_value)
                    elif param_type == ODBType.SYS_REFCURSOR:
                        param_value = cursor.var(oracledb.CURSOR)
                    elif param_type in [ODBType.INT, ODBType.DECIMAL, ODBType.NUMBER,
                                        ODBType.VARCHAR2]:
                        param_value = str(param_value)
                    elif param_type == ODBType.VARCHARLIST:
                        param_value = oracledb.Array(param_value, oracledb.STRING, 32767)
                    elif param_type == ODBType.PSQLCODE:
                        ora_psqlcode["PSQLCODE"] = cursor.var(oracledb.DB_TYPE_VARCHAR)
                        ora_psqlcode["PSQLCODE"].setvalue(0, "0")

                    if param_direction in ["IN"]:
                        send_params.append(param_value)
                    if param_direction in ["OUT", "IN/OUT"]:
                        dynamic_var = ora_psqlcode["PSQLCODE"] if param_type == ODBType.PSQLCODE else param_value
                        send_params.append(dynamic_var)
                        dynamic_vars.append(dynamic_var)
                        dynamic_var_names.append(param_name)
                        dynamic_var_types.append(param_type)

                cursor.callproc(procedure_name, send_params)                
                params_out = get_dynamic_values(dynamic_var_names, dynamic_vars, dynamic_var_types)

                return params_out
        except (oracledb.DatabaseError, oracledb.InterfaceError) as e:
            return {"OOPS": str(e)}
        finally: 
            if self.transaction is False:    
                self.connection.commit()       
                self.close_connection()
                self.disconnect()

    function_mappings = {
        ODBFunctionType.CURSOR: {
            'cursor_type': oracledb.CURSOR,
            'retval_key': 'RETORNO_FUNCION',
            'default_retval': [],
            'process_result': True
        },
        ODBFunctionType.STRING: {
            'cursor_type': oracledb.STRING,
            'retval_key': 'RETORNO_FUNCION',
            'default_retval': None,
            'process_result': False
        },
        ODBFunctionType.NUMBER: {
            'cursor_type': oracledb.NATIVE_INT,
            'retval_key': 'RETORNO_FUNCION',
            'default_retval': None,
            'process_result': False
        }
    }

    def execute_function(self, function_name: str, params: List[dict], type_function: str):
        """
            Ejecuta una función en Oracle con los parámetros proporcionados.

            Args:
                function_name (str): Nombre de la función a ejecutar.
                params (List[dict]): Lista de diccionarios que contienen información de los parámetros.
                    Cada diccionario debe contener las siguientes claves:
                    - "nombre": Nombre del parámetro.
                    - "valor": Valor del parámetro.
                    - "tipo": Tipo de dato del parámetro (debe ser uno de los valores definidos en la clase ODBType).
                    - "direccion": Dirección del parámetro (debe ser uno de los valores: "IN", "OUT", "IN/OUT").
                type_function (str): Tipo de función a ejecutar (debe coincidir con una clave en self.function_mappings).

            Returns:
                Union[dict, None]: Un diccionario con los valores de retorno y los parámetros de salida o None si ocurre un error.

            Raises:
                ConnectionError: Si ocurre un error al crear o establecer la conexión con la base de datos.
        """
        try:
            if self.transaction is False:
                self.create_pool()
                self.connect()
            with self.connection.cursor() as cursor:
                params_function = []
                dynamic_vars = []
                dynamic_var_names = []
                dynamic_var_types = []
                ora_psqlcode = {}
                for param in params:
                    param_name = param["nombre"]
                    param_value = param["valor"]
                    param_type = param["tipo"]
                    param_direction = param["direccion"]

                    if param_type == ODBType.DATE:
                        param_value = oracledb.Timestamp.from_datetime(param_value)
                    elif param_type == ODBType.SYS_REFCURSOR:
                        param_value = cursor.var(oracledb.CURSOR)
                    elif param_type in [ODBType.INT, ODBType.DECIMAL, ODBType.NUMBER,
                                        ODBType.VARCHAR2]:
                        param_value = str(param_value)
                    elif param_type == ODBType.VARCHARLIST:
                        param_value = oracledb.Array(param_value, oracledb.STRING, 32767)
                    elif param_type == ODBType.PSQLCODE:
                        ora_psqlcode["PSQLCODE"] = cursor.var(oracledb.DB_TYPE_VARCHAR)
                        ora_psqlcode["PSQLCODE"].setvalue(0, "0")

                    if param_direction in ["IN"]:
                        params_function.append(param_value)
                    if param_direction in ["OUT", "IN/OUT"]:
                        dynamic_var = ora_psqlcode["PSQLCODE"] if param_type == ODBType.PSQLCODE else param_value
                        params_function.append(dynamic_var)
                        dynamic_vars.append(dynamic_var)
                        dynamic_var_names.append(param_name)
                        dynamic_var_types.append(param_type)

                function_mapping = self.function_mappings.get(type_function)

                if function_mapping:
                    cursor_type = function_mapping['cursor_type']
                    retval_key = function_mapping['retval_key']
                    default_retval = function_mapping['default_retval']
                    process_result = function_mapping['process_result']

                    retval = cursor.callfunc(function_name, cursor_type, params_function)
                    params_out = get_dynamic_values(dynamic_var_names, dynamic_vars, dynamic_var_types)

                    if process_result and retval is not None:
                        cursor_results = process_sys_refcursor(retval)
                    else:
                        cursor_results = default_retval if retval is None else retval

                    return {retval_key: cursor_results, "ARRAY_RET": params_out}
        except (oracledb.DatabaseError, oracledb.InterfaceError) as e:
            return {"OOPS": str(e)}
        finally:
            if self.transaction is False:
                self.connection.commit()
                self.close_connection()
                self.disconnect()

    def execute_query(self, query: str, params: List[dict]):
        """
        Ejecuta una consulta SQL en Oracle con los parámetros proporcionados.

        Args:
            query (str): Consulta SQL a ejecutar.
            params (List[dict]): Lista de diccionarios que contienen información de los parámetros.
                Cada diccionario debe contener las siguientes claves:
                - "nombre": Nombre del parámetro.
                - "valor": Valor del parámetro.
                - "tipo": Tipo de dato del parámetro (debe ser uno de los valores definidos en la clase ODBType).
                - "direccion": Dirección del parámetro (debe ser uno de los valores: "IN", "OUT", "IN/OUT").

        Returns:
            Union[List[dict], None]: Una lista de diccionarios que representa las filas resultantes de la consulta,
            o None si ocurre un error.

        Raises:
            ConnectionError: Si ocurre un error al crear o establecer la conexión con la base de datos.
        """
        try:
            self.create_pool()
            self.connect()

            with self.connection.cursor() as cursor:
                send_params = []
                dynamic_vars = []
                dynamic_var_names = []
                dynamic_var_types = []
                ora_psqlcode = {}

                for param in params:
                    param_name = param["nombre"]
                    param_value = param["valor"]
                    param_type = param["tipo"]
                    param_direction = param["direccion"]

                    if param_type == ODBType.DATE:
                        param_value = oracledb.Timestamp.from_datetime(param_value)
                    elif param_type == ODBType.SYS_REFCURSOR:
                        param_value = cursor.var(oracledb.CURSOR)
                    elif param_type in [ODBType.INT, ODBType.DECIMAL, ODBType.NUMBER, ODBType.VARCHAR2]:
                        param_value = str(param_value)
                    elif param_type == ODBType.VARCHARLIST:
                        param_value = oracledb.Array(param_value, oracledb.STRING, 32767)
                    elif param_type == ODBType.PSQLCODE:
                        ora_psqlcode["PSQLCODE"] = cursor.var(oracledb.DB_TYPE_VARCHAR)
                        ora_psqlcode["PSQLCODE"].setvalue(0, "0")

                    if param_direction in ["IN"]:
                        send_params.append(param_value)
                    if param_direction in ["OUT", "IN/OUT"]:
                        dynamic_var = ora_psqlcode["PSQLCODE"] if param_type == ODBType.PSQLCODE else param_value
                        send_params.append(dynamic_var)
                        dynamic_vars.append(dynamic_var)
                        dynamic_var_names.append(param_name)
                        dynamic_var_types.append(param_type)

                cursor.execute(query, send_params)

                if cursor.description:
                    result = []
                    for row in cursor:
                        row_dict = {}
                        for i, col in enumerate(cursor.description):
                            col_name = col[0]
                            col_value = row[i]
                            if isinstance(col_value, oracledb.LOB):
                                col_value = col_value.read()
                            col_name_camel_case = convert_camel_case(col_name)
                            row_dict[col_name_camel_case] = col_value
                        result.append(row_dict)
                    return result
                else:
                    return None
        except (oracledb.DatabaseError, oracledb.InterfaceError) as e:
            return {"OOPS": str(e)}
        finally:
            self.close_connection()
            self.disconnect()

    def disconnect(self):
        """
        Cierra la conexión del pool de conexiones.

        Returns:
            None
        """
        if self.pool is not None:
            self.pool.close()
            self.pool = None
