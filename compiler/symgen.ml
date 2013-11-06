let sym_num = ref 0;;

let gensym () =
    let sym = "__sym" ^ string_of_int !sym_num ^ "_" in
        sym_num := (!sym_num + 1); sym
