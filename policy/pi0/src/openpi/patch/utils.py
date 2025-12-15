
def dict_equal(d1, d2, atol=1e-6):
    if not (isinstance(d1, dict) and isinstance(d2, dict)):
        return False
    if d1.keys() != d2.keys():
        return False
    for key in d1:
        v1, v2 = d1[key], d2[key]
        if isinstance(v1, dict) and isinstance(v2, dict):
            if not dict_equal(v1, v2, atol=atol):
                return False
        elif isinstance(v1, np.ndarray):
            v2 = np.array(v2)
            v2 = np.squeeze(v2)
            if v1.shape != v2.shape:
                print(v1.shape, v2.shape)
                return False
            if not np.allclose(v1, v2, atol=atol):
                with open("1.txt", "w") as f:
                    for i in range(v1.shape[0]):
                        if not np.allclose(v1[i], v2[i], atol=atol):
                            f.write(str(v1[i]) + "\n")
                            f.write(str(v2[i]) + "\n")
                            print(i)
                            break

                return False
        elif isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
            if len(v1) != len(v2):
                return False
            for elem1, elem2 in zip(v1, v2):
                if not dict_equal(elem1, elem2, atol=atol):
                    return False
        elif v1 != v2:
            return False
    return True

