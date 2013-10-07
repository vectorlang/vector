type binop = Add | Sub | Mul | Div | Mod
    | Lshift | Rshift
    | Less | LessEq | Greater | GreaterEq | Eq | NotEq
    | BitAnd | BitXor | BitOr | LogAnd | LogOr ;;

type preop = Neg | LogNot | BitNot | PreDec | PreInc ;;

type postop = PostDec | PostInc ;;

type expr =
    Binop of expr * binop * expr
  | Preop of preop * expr
  | Postop of expr * postop
  | Assign of string * expr
  | IntLit of int32
  | Int64Lit of int64
  | FloatLit of float
  | ComplexLit of Complex.t
  | StringLit of string
  | CharLit of char
  | Cast of string * expr
  | Ident of string ;;

type statement =
    CompoundStatement of statement list
  | Expression of expr
  | AssigningDecl of string * expr
  | PrimitiveDecl of string * string
  | ArrayDecl of string * string * expr
  | EmptyStatement
