exception Invalid_type of string

type binop = Add | Sub | Mul | Div | Mod
    | Lshift | Rshift
    | Less | LessEq | Greater | GreaterEq | Eq | NotEq
    | BitAnd | BitXor | BitOr | LogAnd | LogOr

type unop = Neg | LogNot | BitNot

type postop = Dec | Inc

type datatype =
    Bool | Char | Int8
  | UInt8
  | Int16
  | UInt16
  | Int32
  | UInt32
  | Int64
  | UInt64
  | Float32
  | Float64
  | Complex64
  | Complex128
  | String
  | Void
  | ArrayType of datatype

type ident =
    Ident of string

type lvalue =
  | Variable of ident
  | ArrayElem of ident * expr list
  | ComplexAccess of expr * ident 
and expr =
    Binop of expr * binop * expr
  | AssignOp of lvalue * binop * expr
  | Unop of unop * expr
  | PostOp of lvalue * postop
  | Assign of lvalue * expr
  | IntLit of int32
  | Int64Lit of int64
  | FloatLit of float
  | ComplexLit of expr * expr
  | StringLit of string
  | CharLit of char
  | ArrayLit of expr list
  | Cast of datatype * expr
  | FunctionCall of ident * expr list
  | HigherOrderFunctionCall of ident * ident * expr
  | Lval of lvalue

type decl =
    AssigningDecl of ident * expr
  | PrimitiveDecl of datatype * ident
  | ArrayDecl of datatype * ident * expr list

type range =
    Range of expr * expr * expr

type iterator =
    RangeIterator of ident * range
  | ArrayIterator of ident * expr

type statement =
    CompoundStatement of statement list
  | Declaration of decl
  | Expression of expr
  | EmptyStatement
  | IfStatement of expr * statement * statement
  | WhileStatement of expr * statement
  | ForStatement of iterator list * statement
  | PforStatement of iterator list * statement
  | FunctionDecl of bool * datatype * ident * decl list * statement list
  | ForwardDecl of bool * datatype * ident * decl list
  | ReturnStatement of expr
  | VoidReturnStatement

let type_of_string = function
    | "bool" -> Bool
    | "char" -> Char
    | "int8" -> Int8
    | "uint8" -> UInt8
    | "int16" -> Int16
    | "uint16" -> UInt16
    | "int" -> Int32
    | "int32" -> Int32
    | "uint" -> UInt32
    | "uint32" -> UInt32
    | "int64" -> Int64
    | "uint64" -> UInt64
    | "double" -> Float64
    | "float" -> Float32
    | "float32" -> Float32
    | "float64" -> Float64
    | "complex" -> Complex64
    | "complex64" -> Complex64
    | "complex128" -> Complex128
    | "void" -> Void
    | dtype -> raise (Invalid_type dtype)
